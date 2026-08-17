"""Reformulation engine for the Hermes prompt watcher.

Detects when a message should be rewritten, protects JSON/code blocks so they
travel through the reformulation model untouched, and calls the upstream
gateway (OmniRoute) with the configured combo model.
"""

import asyncio
import json
import logging
import re

log = logging.getLogger("watcher.reformulate")

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
TRIPLE_BRACKET_RE = re.compile(r"\[\[\[.*?\]\]\]", re.DOTALL)
PLACEHOLDER = "{{BLOCK_%d}}"
PLACEHOLDER_RE = re.compile(r"\{\{BLOCK_(\d+)\}\}")
SYSTEM_MARKER_RE = re.compile(r"^\[System:[^\]]*\]\s*", re.IGNORECASE)
TECH_LINE_RE = re.compile(
    r"(?:@\s|://|\.\w{1,6}:\d+|\(anonymous\)|https?://"
    r"|\d{2}:\d{2}:\d{2}|req=|sess=|\[(?:API|MATCHES|AGENT|SOFASCORE|OPENCODE_GO|OMNIROUTE|ORCHESTRATOR|GoalsModelV2|APIFY|UEFA)[^\]]*\]"
    r"|\"GET |\"POST |\"OPTIONS |\"PUT |\"DELETE |\bnet::ERR_|\bUncaught \w+:|\bat \w+ \(.*\.\w+:\d+\)"
    r"|\bat \w+(?:\.\w+)* \(\w+:\d+:\d+\)|\bat \w+ [A-Za-z]+://)"
)


def strip_system_marker(message: str) -> tuple[str, str]:
    """Split a leading '[System: ...]' Hermes marker from the rest of the message.

    Hermes prefixes the next user message with '[System: The active model ...]'
    after a model/provider switch. The marker must travel to the model untouched;
    only the trailing user text may be reformulated.
    Returns (marker, remainder); marker is '' when absent.
    """
    m = SYSTEM_MARKER_RE.match(message)
    if not m:
        return "", message
    return m.group(0), message[m.end():]

SYSTEM_PROMPT = (
    "Tu es un expert en prompt engineering pour agents IA, avec 15+ ans d'expérience en "
    "ingénierie logicielle, debug et architecture.\n"
    "Ton unique tâche : transformer le message brut de l'utilisateur en un prompt OPTIMAL "
    "— clair, structuré, actionnable, sans ambiguïté — qui donne à l'agent cible toutes les "
    "chances de réussir du premier coup. Tu améliores, tu ne te contentes pas de corriger.\n\n"
    "PROCESSUS :\n"
    "1. Identifie l'intention réelle et toutes les informations (faits, contraintes, livrables, contexte technique).\n"
    "2. Détecte le TYPE de tâche et applique les meilleures pratiques correspondantes :\n"
    "   - Debug/bug : demande d'analyser la cause racine AVANT de fixer, cite les fichiers/lignes, "
    "     demande les logs/erreurs nécessaires si absents, liste les hypothèses puis la vérification.\n"
    "   - Feature/développement : précise le comportement attendu, les contraintes (perf, style, "
    "     compat), les cas limites, la vérification finale (tests, build).\n"
    "   - Refactor : demande de préserver le comportement, de respecter les conventions existantes.\n"
    "   - Exploration/explication : demande une synthèse structurée avec références aux fichiers.\n"
    "   - Décision/conseil : demande les options, leurs trade-offs, puis une recommandation.\n"
    "   - Tâche vague : reformule en énonçant explicitement les hypothèses raisonnables et ce qui "
    "     est attendu en sortie.\n"
    "3. Structure le résultat : ordre logique (contexte → tâche → contraintes → livrable → vérification).\n"
    "4. Enrichis : ajoute les précisions utiles qui manquent (format de sortie, périmètre, critères "
    "d'acceptation, actions concrètes demandées à l'agent) — uniquement des améliorations utiles, "
    "jamais d'invention de faits, de chiffres, de données ou d'infos techniques absentes.\n"
    "5. Corrige l'orthographe, la grammaire et les fautes de frappe.\n\n"
    "RÈGLES STRICTES :\n"
    "- CONSERVE chaque information : ne retire rien, n'invente rien, ne déforme pas l'intention.\n"
    "- NE TRADUIS JAMAIS : garde la langue d'origine (français reste français, anglais reste anglais).\n"
    "- Ne colle JAMAIS des termes techniques, noms de fichiers, fonctions, bibliothèques ou technologies : "
    "conserve-les EXACTEMENT comme écrits (casse, accents, extensions) — ne les 'corrige' jamais.\n"
    "- Sois direct et opérationnel : la demande doit être exécutable par un agent sans question de clarification.\n"
    "- Si le message contient des placeholders {{BLOCK_n}}, conserve-les tels quels à leur position : "
    "ce sont des données qui seront réinsérées telles quelles, à ne jamais reformuler.\n"
    "- IMPORTANT : les placeholders {{BLOCK_n}} ne t'exemptent PAS de reformuler et améliorer le reste. "
    "La prose qui les entoure (avant, entre, après) DOIT être reformulée et améliorée normalement. "
    "Seuls les placeholders restent intacts, à leur position exacte.\n"
    "- Chaque placeholder {{BLOCK_n}} représente un segment que l'utilisateur a protégé avec [[[ ]]] "
    "(triples crochets) ou un bloc collé (code, log) : conserve-le TOUJOURS tel quel, à sa position, "
    "sans le déplacer, le reformuler ni le résumer.\n"
    "- NE RÉPONDS PAS à la demande de l'utilisateur : renvoie uniquement le prompt amélioré, sans préambule, "
    "sans guillemets, sans commentaire, sans liste de changements, sans explication.\n\n"
    "ERREURS DE FRAPPE RAPIDE (l'utilisateur tape vite, sans se relire) :\n"
    "- Espace manquant entre deux mots : 'outa' → 'ou là', 'jemapelle' → 'je m'appelle', "
    "'lextension' → 'l'extension'.\n"
    "- Lettres manquantes, inversées ou doublées : 'ocmbiné' → 'combiné', 'deter' → 'déterrer', "
    "'qulque' → 'quelque'.\n"
    "- Ne transforme JAMAIS un mot inconnu ou suspect en acronyme ou en nom propre : si un mot "
    "n'est pas un acronyme connu dans le contexte, c'est une faute de frappe — reconstruis le mot "
    "français le plus plausible.\n"
    "- Utilise le sens global de la phrase pour deviner le mot voulu ; en cas de doute, "
    "garde la correction la plus naturelle et la plus simple.\n\n"
    "EXEMPLES :\n"
    "Message : 'salut jveux un script qui trie les fichiers par type et apres les deplacer dans des dossiers stp'\n"
    "Reformulation attendue : 'Crée un script qui trie les fichiers d'un dossier par type (extension), "
    "puis déplace chaque fichier dans le sous-dossier correspondant. Précise le langage, le dossier "
    "source, et gère le cas où le sous-dossier n'existe pas (le créer). Vérifie le résultat.'\n\n"
    "Message : 'le truc marche plus apres ma modif y a une erreur react'\n"
    "Reformulation attendue : 'Après ma modification, une erreur React apparaît et la fonctionnalité "
    "ne fonctionne plus. Analyse la cause racine en comparant avec l'état précédent, identifie le "
    "fichier et la ligne fautifs, explique pourquoi, puis propose et applique le fix minimal. "
    "Vérifie que rien d'autre n'est cassé.'\n\n"
    "Message : 'fait une disterde de 2 ligne stp'\n"
    "Reformulation attendue : 'Rédige une dissertation de deux lignes, s'il te plaît.'\n\n"
    "Message : 'explique mwa en detlis'\n"
    "Reformulation attendue : 'Explique-moi en détail.'\n\n"
    "Message : 'est ce que je peux faire le mode agent ocmbiné o uta des truc a changé niveau front ?'\n"
    "Reformulation attendue : 'Est-ce que je peux faire le mode agent combiné ou alors avec des trucs qui ont changé au niveau du front-end ?'\n\n"
    "Message : 'continue ducoup l'erreur entiere : {{BLOCK_0}}'\n"
    "Reformulation attendue : 'Continue. Voici l'erreur entière : {{BLOCK_0}}. Analyse la cause racine "
    "(fichiers et lignes cités dans la stack trace), propose le fix, puis applique-le.'"
)


def extract_blocks(text: str) -> tuple[str, list[str]]:
    """Replace protected segments with {{BLOCK_n}} placeholders, in order:

    1. Fenced code blocks (``` ... ```).
    2. Triple-bracket segments ([[[ ... ]]]) — the user's explicit "keep this
       verbatim" syntax: everything inside [[[ ]]] is sent back exactly as-is,
       only the prose outside is reformulated.
    3. Runs of technical lines (stack traces, logs, source references) so
       pasted diagnostics travel through reformulation untouched.

    A technical run needs at least 2 consecutive technical lines; surrounding
    prose is left for the model.
    """
    blocks: list[str] = []

    def repl(m: re.Match) -> str:
        blocks.append(m.group(0))
        return PLACEHOLDER % (len(blocks) - 1)

    text = FENCE_RE.sub(repl, text)
    text = TRIPLE_BRACKET_RE.sub(repl, text)

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if TECH_LINE_RE.search(lines[i]):
            j = i + 1
            while j < n and (TECH_LINE_RE.search(lines[j]) or _is_short_tail(lines, j)):
                j += 1
            if j - i >= 2:
                blocks.append("\n".join(lines[i:j]))
                out.append(PLACEHOLDER % (len(blocks) - 1))
                i = j
                continue
        out.append(lines[i])
        i += 1

    return "\n".join(out), blocks


def _is_short_tail(lines: list[str], idx: int) -> bool:
    """Absorb short non-technical lines (e.g. 'postMessage', 'setTimeout')
    into a technical run so stack traces stay contiguous."""
    line = lines[idx].strip()
    if not line or len(line) > 40:
        return False
    if TECH_LINE_RE.search(lines[idx]):
        return True
    nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
    return bool(TECH_LINE_RE.search(nxt))


def restore_blocks(text: str, blocks: list[str]) -> str:
    """Restore {{BLOCK_n}} placeholders with the original block content.

    Iterates until stable because a block can itself contain a placeholder
    (e.g. a [..] marker embedded inside a protected technical run).
    """

    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        return blocks[i] if 0 <= i < len(blocks) else m.group(0)

    result = text
    for _ in range(len(blocks) + 2):
        new = PLACEHOLDER_RE.sub(repl, result)
        if new == result:
            break
        result = new
    return result


def is_pure_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
        return True
    except (ValueError, TypeError):
        return False


def is_large_block(text: str, max_chars: int = 5000, max_lines: int = 60) -> bool:
    """True when the message looks like a copied/pasted block (log, dump,
    config, code without fences) that must travel intact — never reformulated.

    Heuristic: total length above max_chars, or many lines, or at least one
    very long line (e.g. a minified/JSON blob or a stack trace line).
    """
    if len(text) > max_chars:
        return True
    lines = text.splitlines()
    if len(lines) > max_lines:
        return True
    return any(len(line) > 500 for line in lines)


def has_tech_run(text: str, min_run: int = 2) -> bool:
    """True when at least `min_run` consecutive lines look technical."""
    run = 0
    for line in text.splitlines():
        if TECH_LINE_RE.search(line):
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


DEFAULT_SKIP_PREFIXES = (
    "[System",
    "[IMPORTANT",
    "You are a summarization agent",
    "You are updating a context compaction summary",
    "Create a structured checkpoint summary",
)


def decide(
    message: str,
    min_words: int,
    skip_prefixes=DEFAULT_SKIP_PREFIXES,
    max_chars: int = 5000,
    max_lines: int = 60,
) -> str:
    """Return 'passthrough' | 'protect' | 'reformulate'."""
    if not message or not message.strip():
        return "passthrough"
    stripped = message.lstrip()
    if any(stripped.startswith(p) for p in skip_prefixes):
        return "passthrough"
    if word_count(message) < min_words:
        return "passthrough"
    if is_pure_json(message):
        return "passthrough"
    if FENCE_RE.search(message) or TRIPLE_BRACKET_RE.search(message) or has_tech_run(message):
        # Pasted blocks detected (fenced, [[[ ]]] user markers, or stack-trace/
        # log runs): protect the blocks as placeholders and reformulate only the
        # surrounding prose. This applies even for big messages — a log with
        # [API] prefixes is content, not protection syntax, but its lines are
        # technical runs and get protected as whole blocks.
        protected, _ = extract_blocks(message)
        prose = PLACEHOLDER_RE.sub("", protected)
        if word_count(prose) >= min_words:
            return "protect"
        # The message is almost entirely one pasted block with no prose to
        # reformulate around it -> send it intact.
        return "passthrough"
    if is_large_block(message, max_chars, max_lines):
        # Big blob of pure prose with no identifiable blocks: keep intact.
        return "passthrough"
    return "reformulate"


def build_user_prompt(message: str) -> str:
    return "Reformule le message suivant :\n\n" + message


async def call_upstream(message: str, combo_model: str, upstream_url: str, api_key: str,
                        timeout_s: float, session) -> str | None:
    """One-shot non-streaming reformulation call to the upstream gateway."""
    payload = {
        "model": combo_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(message)},
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 16000,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with session.post(
            upstream_url.rstrip("/") + "/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_s,
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                log.warning("reformulation upstream status %s: %.300s", resp.status, body)
                return None
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                return None
            return content.strip()
    except (asyncio.TimeoutError, Exception) as exc:
        log.warning("reformulation failed: %s", exc)
        return None
