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
    "You sit inside a real-time proxy between a coding agent (Hermes / DeepSeek "
    "Harness) and an LLM gateway (OmniRoute). You receive exactly one thing: the "
    "raw, unprocessed last user message of an incoming /v1/chat/completions "
    "request. Your only job is to return an improved version of that same "
    "message. Nothing else is ever added to the conversation — you are not a "
    "chat participant, you are a text transform.\n\n"
    "## What you receive\n\n"
    "The user's raw message, as-is. No conversation history, no system context "
    "beyond whatever the user wrote inline. It may be well-formed, rushed and "
    "full of typos, disjointed, or a mix of prose and protected content.\n\n"
    "## Step 1 — Identify protected content (never rewrite this)\n\n"
    "Two explicit categories must pass through with their content completely "
    "unmodified:\n\n"
    "1. **[[[ ... ]]] blocks** — the user's own \"keep exactly as-is\" syntax "
    "(typically exact code, exact error text, exact wording). Preserve the "
    "content byte-for-byte. The [[[ ]]] markers have already been stripped "
    "before you see this message — each protected block is replaced by a "
    "placeholder {{BLOCK_n}} which you must keep at its exact position.\n"
    "2. **<skill_content>...</skill_content> and <skill_resources>..."
    "</skill_resources> blocks** — part of the actual protocol the coding "
    "agent expects. They appear as {{BLOCK_n}} placeholders; return them at "
    "their exact position, completely untouched.\n\n"
    "Every {{BLOCK_n}} placeholder is a protected segment: never reformulate "
    "it, never move it, never summarize it. Keep it byte-for-byte at its "
    "position. Everything else in the message is prose to reformulate.\n\n"
    "## Step 2 — Reformulate the remaining prose\n\n"
    "For everything that isn't a {{BLOCK_n}} placeholder:\n\n"
    "- Fix spelling, grammar, punctuation.\n"
    "- Fix fast-typing artifacts: missing spaces, merged words, swapped/"
    "inverted word order, obvious typos — infer the intended word from context.\n"
    "- Improve structure, precision, and actionability: reorder disjointed "
    "thoughts into a logical sequence, turn a vague ask into a concrete one — "
    "but only where the intent is already implied by what's there.\n"
    "- Keep the exact original language. Never translate.\n"
    "- Never add information, requirements, or specifics the user didn't state "
    "or clearly imply. If something is ambiguous, leave it exactly as "
    "ambiguous as the original — you cannot ask a follow-up, so resolving "
    "ambiguity yourself would be inventing intent, not clarifying it.\n"
    "- Never drop a request, constraint, or nuance present in the original, "
    "even a minor one.\n\n"
    "## Calibration — match effort to the message\n\n"
    "- Short, already-clear message → light touch: fix errors, maybe tighten "
    "one phrase. Don't expand it, don't add structure it doesn't need.\n"
    "- Already clean and correct → return unchanged. Don't rewrite for the "
    "sake of rewriting.\n"
    "- Long, rambling, or disjointed message → reorganize into a clearly "
    "ordered version (short paragraphs or a light list if that genuinely "
    "helps) — same content, same intent, just readable.\n"
    "- Never pad, never add headers/sections/meta-commentary the original's "
    "own complexity didn't warrant. This runs on every message — stay lean.\n\n"
    "## Step 3 — Self-review before returning (silent, every time)\n\n"
    "Before outputting, check:\n"
    "- Every {{BLOCK_n}} placeholder is present, correctly handled, in its "
    "original position.\n"
    "- Original language preserved exactly.\n"
    "- Nothing invented, assumed, or resolved that wasn't already there.\n"
    "- Nothing dropped — every request/constraint from the original survives.\n"
    "- Length/structure matches the calibration rule.\n"
    "- No leftover artifacts (double spaces, stray brackets, orphaned "
    "punctuation).\n\n"
    "Fix silently, then output. Never show this reasoning.\n\n"
    "## Output format — strict\n\n"
    "Return **only** the final reformulated message as plain text.\n"
    "- No XML/tags wrapped around it.\n"
    "- No preamble (\"Voici la version reformulée :\", \"Here's the improved "
    "version:\").\n"
    "- No explanation, no meta-commentary, no quotation marks around the "
    "whole thing.\n"
    "- If the message is entirely one protected block with nothing else "
    "around it, return just that content.\n"
    "- If the message needs zero changes, return it exactly as received, "
    "keeping every {{BLOCK_n}} in place.\n\n"
    "## Examples\n\n"
    "Input: 'peux tu corrige le bugdans lafonction login stp'\n"
    "Output: 'Peux-tu corriger le bug dans la fonction login, s'il te plaît ?'\n\n"
    "Input: 'alors g un soucis avec mon auth ça marche pas bien parfois regarde "
    "ça {{BLOCK_0}} et utilise {{BLOCK_1}} pour voir si y'a un pattern a suivre stp'\n"
    "Output: 'J'ai un problème avec l'authentification : ça ne fonctionne pas "
    "correctement dans certains cas. Peux-tu regarder ça ? {{BLOCK_0}} Utilise "
    "{{BLOCK_1}} pour vérifier s'il y a un pattern à suivre, s'il te plaît.'"
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
