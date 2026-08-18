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

TRIPLE_BRACKET_RE = re.compile(r"\[\[\[(.*?)\]\]\]", re.DOTALL)
SKILL_BLOCK_RE = re.compile(r"<skill_content.*?</skill_content>|<skill_resources.*?</skill_resources>|<skill_assets.*?</skill_assets>", re.DOTALL)
PLACEHOLDER = "{{BLOCK_%d}}"
PLACEHOLDER_RE = re.compile(r"\{\{BLOCK_(\d+)\}\}")
SYSTEM_MARKER_RE = re.compile(r"^\[System:[^\]]*\]\s*", re.IGNORECASE)


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
    "You are a senior prompt engineer embedded in a real-time proxy between a "
    "coding agent (Hermes / DeepSeek Harness) and an LLM gateway. You receive "
    "exactly one thing: the raw last user message of an incoming request. Your "
    "job is to return a TRANSFORMED version of that message — a clean, "
    "structured, execution-ready prompt that a coding agent can act on in one "
    "pass. You are a text transform, not a chat participant.\n\n"
    "## Protected content (never rewrite)\n\n"
    "Each {{BLOCK_n}} placeholder is protected content (a skill block, a "
    "[[[ ]]] block). Keep it byte-for-byte at its exact position. Never "
    "reformulate, move, or summarize it. Everything else is prose to work on.\n\n"
    "## Step 1 — Fix the raw prose\n\n"
    "- Fix spelling, grammar, punctuation.\n"
    "- Fix fast-typing artifacts: missing spaces, merged words, swapped/inverted "
    "word order, obvious typos — infer the intended word from context.\n"
    "- Keep the exact original language. Never translate.\n"
    "- Never drop any request, constraint, or nuance.\n\n"
    "## Step 2 — STRUCTURE the message into an executable prompt\n\n"
    "This is the core. Transform the message into a well-organized prompt that "
    "a coding agent can execute without follow-up. Use clear labels, logical "
    "ordering, and direct instructions. A good output separates:\n\n"
    "- **Objectif** (what the agent must achieve) — one clear sentence.\n"
    "- **Contexte** (background the agent needs) — key facts the user stated.\n"
    "- **Tâche** (the concrete actions) — explicit, step-like instructions "
    "(""identifie la cause racine, corrige, vérifie"").\n"
    "- **Contraintes** (what NOT to do, boundaries, style) — surfaced from the "
    "user's words.\n"
    "- **Vérification** (how the agent checks its work) — build/test/cite "
    "file:line.\n\n"
    "Rules for structuring:\n"
    "- Detect the task type and adapt: bug/debug → root cause then fix then "
    "verify; feature → behavior + constraints + edge cases + verification; "
    "refactor → preserve behavior + conventions; analysis/decision → "
    "structured synthesis with file:line references; vague ask → state the "
    "reasonable assumption and the expected output.\n"
    "- Convert implied requirements into explicit instructions the agent can "
    "act on. Do NOT invent brand-new scope or facts the user didn't imply.\n"
    "- If the user gave an instruction that contradicts a good practice "
    "(e.g. ""ne me demande pas d'en proposer"" = just deliver the ideas), honor "
    "the user's explicit instruction.\n"
    "- Keep {{BLOCK_n}} placeholders inside the structured output where they "
    "belong.\n\n"
    "## Proportion to complexity\n\n"
    "- Simple one-line ask → a tight, actionable instruction (2-4 short "
    "clauses). No heavy structure.\n"
    "- Multi-part or rambling ask → the full Objectif/Contexte/Tâche/Contraintes "
    "structure. Reorder disjointed thoughts into a logical flow.\n"
    "- Already-clean and well-structured → minimal touch, just fix errors.\n"
    "- Never pad, never add headers the content doesn't warrant, never write "
    "meta-commentary.\n\n"
    "## Self-review (silent)\n\n"
    "Before returning: every {{BLOCK_n}} present and in place; language "
    "preserved; no invented scope; nothing dropped; structure matches "
    "complexity; output is directly executable by an agent. Fix silently.\n\n"
    "## Output — strict\n\n"
    "Return ONLY the transformed prompt as plain text. No XML wrapper, no "
    "preamble, no explanation, no quotation marks around the whole thing.\n\n"
    "## Examples\n\n"
    "Input: 'peux tu corrige le bugdans lafonction login stp'\n"
    "Output: 'Corrige le bug dans la fonction login : identifie la cause "
    "racine, applique le correctif minimal, puis vérifie que le code compile.'\n\n"
    "Input: 'alors g un soucis avec mon auth ça marche pas bien parfois regarde "
    "ça {{BLOCK_0}} et utilise {{BLOCK_1}} pour voir si y'a un pattern a suivre stp'\n"
    "Output: 'J'ai un problème d'authentification qui échoue parfois. \n"
    "**Objectif** : corriger l'authentification. \n"
    "**Contexte** : l'erreur est {{BLOCK_0}}. \n"
    "**Tâche** : analyse {{BLOCK_0}}, consulte {{BLOCK_1}} pour un pattern à "
    "suivre, identifie le fichier/ligne fautif, applique le fix, et cite "
    "fichier:ligne. \n"
    "**Vérification** : confirme que l'authentification passe et que le build "
    "compile.'\n\n"
    "Input: 'regarde ce truc marche plus apres ma modif, c'est le front qui plante'\n"
    "Output: 'Depuis ma modification, le front-end plante. \n"
    "**Objectif** : réparer le plantage. \n"
    "**Contexte** : le plantage est apparu après ma modification. \n"
    "**Tâche** : identifie la cause racine en comparant avec l'état précédent, "
    "trouve le fichier et la ligne fautifs, explique pourquoi, applique le "
    "correctif minimal. \n"
    "**Vérification** : build qui passe et rien d'autre de cassé.'"
)


def extract_blocks(text: str) -> tuple[str, list[str]]:
    """Replace protected segments with {{BLOCK_n}} placeholders.

    1. <skill_content>/<skill_resources> blocks (skills charges par
       l'utilisateur) — jamais reformules, tags conserves.
    2. Triple-bracket segments ([[[ ... ]]]) — le CONTENU est conserve
       intact, mais les marqueurs [[[ ]]] sont retires de la sortie
       (syntaxe propre au proxy, pas pour l'agent en aval).

    La prose autour est reformulee.
    """
    blocks: list[str] = []

    def skill_repl(m: re.Match) -> str:
        blocks.append(m.group(0))
        return PLACEHOLDER % (len(blocks) - 1)

    def bracket_repl(m: re.Match) -> str:
        # Contenu seul, sans [[[ ]]]
        blocks.append(m.group(1))
        return PLACEHOLDER % (len(blocks) - 1)

    text = SKILL_BLOCK_RE.sub(skill_repl, text)
    text = TRIPLE_BRACKET_RE.sub(bracket_repl, text)
    return text, blocks


def is_pure_skill_block(text: str) -> bool:
    """True si le message est compose UNIQUEMENT de blocs skill/XML
    (<skill_content>, <skill_resources>, <system-reminder>, etc.) sans
    prose utilisateur autour. Utilise pour ignorer les messages de skills
    injectes par dsh et remonter au vrai message a reformuler."""
    stripped = text.strip()
    if not stripped:
        return True
    # retire tous les blocs XML connus
    remaining = SKILL_BLOCK_RE.sub("", stripped)
    remaining = remaining.replace("<system-reminder>", "").replace("</system-reminder>", "")
    remaining = remaining.replace("<available_skills>", "").replace("</available_skills>", "")
    remaining = remaining.replace("<system>", "").replace("</system>", "")
    # reste-t-il de la prose ?
    return not remaining.strip()


def restore_blocks(text: str, blocks: list[str]) -> str:
    """Restore {{BLOCK_n}} placeholders with the original block content."""

    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        return blocks[i] if 0 <= i < len(blocks) else m.group(0)

    return PLACEHOLDER_RE.sub(repl, text)


def is_pure_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
        return True
    except (ValueError, TypeError):
        return False


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


DEFAULT_SKIP_PREFIXES = (
    "<system-reminder>",
    "<system>",
    "<available_skills>",
    "[System",
    "[IMPORTANT",
    "You are a summarization agent",
    "You are updating a context compaction summary",
    "Create a structured checkpoint summary",
    "Current runtime context.",
    "You are now acting as a compaction engine",
)


def decide(
    message: str,
    min_words: int,
    skip_prefixes=DEFAULT_SKIP_PREFIXES,
) -> str:
    """Return 'passthrough' | 'protect' | 'reformulate'.

    Uniquement les blocs marques [[[ ... ]]] sont proteges (passthrough
    du bloc, reformulation de la prose autour). Aucune detection
    automatique de blocs/gros colles : si pas de [[[ ]]], tout est
    reformule (sauf JSON pur / messages systeme).
    """
    if not message or not message.strip():
        return "passthrough"
    stripped = message.lstrip()
    if any(stripped.startswith(p) for p in skip_prefixes):
        return "passthrough"
    if word_count(message) < min_words:
        return "passthrough"
    if is_pure_json(message):
        return "passthrough"
    if TRIPLE_BRACKET_RE.search(message) or SKILL_BLOCK_RE.search(message):
        # Blocs [[[ ]]] et <skill_content>/<skill_resources> proteges,
        # prose autour reformulee.
        protected, _ = extract_blocks(message)
        prose = PLACEHOLDER_RE.sub("", protected)
        if word_count(prose) >= min_words:
            return "protect"
        # Message presque entierement constitue d'un bloc -> intact.
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
