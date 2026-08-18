"""Hermes prompt watcher.

OpenAI-compatible reverse proxy that sits between Hermes and the model gateway
(OmniRoute on localhost:20128 by default). Every /v1/chat/completions request is
inspected: the last plain-text user message is reformulated via the configured
OmniRoute combo so Hermes always receives a clean, promptable instruction.

- JSON payloads and fenced code blocks are never rewritten (passthrough).
- Any failure in the reformulation is fail-open: the original message is forwarded.
- Streaming (SSE) is relayed byte-for-byte.
- Non-chat endpoints (e.g. /v1/models) are relayed untouched.

Usage:
    python watcher.py [--listen 127.0.0.1:20500] [--upstream http://localhost:20128]
                      [--combo-model NAME] [--api-key KEY] [--min-words 4]

Settings are read, in precedence order: CLI args > watcher.env (KEY=VALUE lines
next to this script) > environment variables > built-in defaults.
Env vars: WATCHER_LISTEN, WATCHER_UPSTREAM, WATCHER_COMBO_MODEL, WATCHER_API_KEY,
WATCHER_MIN_WORDS, WATCHER_TIMEOUT.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import ClientSession, ClientTimeout, web

import reformulate

log = logging.getLogger("watcher")

DEFAULTS = {
    "listen": "127.0.0.1:20500",
    "upstream": "http://localhost:20128",
    "combo_model": "",
    "api_key": "",
    "min_words": 4,
    "timeout": 30,
    "show_rewrite": "1",
    "cache_ttl": "600",
    "skip_prefixes": "<system-reminder>,<system>,<available_skills>,<System,[IMPORTANT,You are a summarization agent,You are updating a context compaction summary,Create a structured checkpoint summary,Current runtime context.,You are now acting as a compaction engine",
}

ENV_MAP = {
    "WATCHER_LISTEN": "listen",
    "WATCHER_UPSTREAM": "upstream",
    "WATCHER_COMBO_MODEL": "combo_model",
    "WATCHER_API_KEY": "api_key",
    "WATCHER_MIN_WORDS": "min_words",
    "WATCHER_TIMEOUT": "timeout",
    "WATCHER_SHOW_REWRITE": "show_rewrite",
    "WATCHER_CACHE_TTL": "cache_ttl",
    "WATCHER_SKIP_PREFIXES": "skip_prefixes",
}


def load_env_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def build_config(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Hermes prompt watcher proxy")
    parser.add_argument("--listen", help="listen address, e.g. 127.0.0.1:20500")
    parser.add_argument("--upstream", help="upstream gateway base URL, e.g. http://localhost:20128")
    parser.add_argument("--combo-model", help="OmniRoute combo model name used for reformulation")
    parser.add_argument("--api-key", help="upstream API key (default: forward the client's key)")
    parser.add_argument("--min-words", type=int, help="skip reformulation for messages shorter than this")
    parser.add_argument("--timeout", type=float, help="reformulation timeout in seconds")
    args = parser.parse_args(argv)

    cfg = dict(DEFAULTS)
    cfg.update(os.environ)  # fill in any WATCHER_* set in the shell

    for env_key, cfg_key in ENV_MAP.items():
        if cfg.get(env_key):
            cfg[cfg_key] = cfg[env_key]

    file_vals = load_env_file(Path(__file__).with_name("watcher.env"))
    for env_key, cfg_key in ENV_MAP.items():
        if env_key in file_vals:
            cfg[cfg_key] = file_vals[env_key]

    overrides = {
        "listen": args.listen,
        "upstream": args.upstream,
        "combo_model": args.combo_model,
        "api_key": args.api_key,
        "min_words": args.min_words,
        "timeout": args.timeout,
        "show_rewrite": None,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val

    cfg["min_words"] = int(cfg["min_words"])
    cfg["timeout"] = float(cfg["timeout"])
    cfg["cache_ttl"] = float(cfg.get("cache_ttl", "600"))
    cfg["show_rewrite"] = str(cfg.get("show_rewrite", "1")).lower() in ("1", "true", "yes", "on")
    cfg["skip_prefixes"] = tuple(
        p.strip() for p in str(cfg.get(
            "skip_prefixes",
            "[System,[IMPORTANT,You are a summarization agent,"
            "You are updating a context compaction summary,"
            "Create a structured checkpoint summary",
        )).split(",") if p.strip()
    )
    return cfg


class Counters:
    def __init__(self):
        self.total = 0
        self.reformulated = 0
        self.protected = 0
        self.skipped = 0
        self.failed = 0

    def snapshot(self) -> dict:
        return {
            "total": self.total,
            "reformulated": self.reformulated,
            "protected": self.protected,
            "skipped": self.skipped,
            "failed": self.failed,
        }


class WatcherApp:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.counters = Counters()
        self.session: ClientSession | None = None
        self.cache: dict[str, tuple[float, str]] = {}
        self.cache_hits = 0

    async def start(self) -> None:
        self.session = ClientSession(timeout=ClientTimeout(connect=20, sock_connect=20))

    async def stop(self) -> None:
        if self.session:
            await self.session.close()

    def upstream_url(self, path: str) -> str:
        return self.cfg["upstream"].rstrip("/") + path

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "upstream": self.cfg["upstream"],
                "combo_model": self.cfg["combo_model"],
                "counters": self.counters.snapshot(),
                "cache_hits": self.cache_hits,
            }
        )

    async def relay(self, request: web.Request, data=None) -> web.StreamResponse:
        """Passthrough for any endpoint we do not special-case (e.g. /v1/models)."""
        url = self.upstream_url(request.path_qs)
        headers = self._outbound_headers(request)
        try:
            upstream_resp = await self.session.request(
                request.method, url, headers=headers,
                data=data if data is not None else request.content,
                allow_redirects=False,
            )
        except Exception as exc:
            log.error("relay %s failed: %s", url, exc)
            return web.json_response({"error": {"message": f"watcher relay failed: {exc}"}}, status=502)
        return await self._relay_response(upstream_resp, request)

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        self.counters.total += 1
        raw = await request.read()
        try:
            body = json.loads(raw)
        except (ValueError, TypeError):
            log.warning("invalid JSON body; passthrough")
            return await self.relay(request, data=raw)

        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return await self.relay(request, data=raw)

        orig_model = body.get("model", "")
        msgs = body["messages"]
        last = None
        for msg in reversed(msgs):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                last = msg
                break

        if last is None:
            log.info("no plain-text user message; passthrough (model=%s)", orig_model)
            return await self.relay(request, data=raw)

        original = last["content"]
        marker, body_text = reformulate.strip_system_marker(original)
        decision = reformulate.decide(
            body_text,
            self.cfg["min_words"],
            self.cfg["skip_prefixes"],
        )

        if decision == "passthrough":
            self.counters.skipped += 1
            stripped = body_text.lstrip()
            if any(stripped.startswith(p) for p in self.cfg["skip_prefixes"]):
                reason = "message interne"
            elif reformulate.is_pure_json(body_text):
                reason = "message JSON pur"
            elif reformulate.word_count(body_text) < self.cfg["min_words"]:
                reason = "trop court"
            else:
                reason = "vide"
            log.info("[model=%s] passthrough (%s): %.80r", orig_model, reason, original)
            return await self.relay(request, data=raw)

        if not self.cfg["combo_model"]:
            self.counters.skipped += 1
            log.info("[model=%s] passthrough (combo_model non configure): %.80r", orig_model, original)
            return await self.relay(request, data=raw)

        started = time.monotonic()
        cached = self._cache_get(original)
        if cached is not None:
            self.cache_hits += 1
            result = cached
            log.info("[model=%s] %s (cache): %.80r -> %.80r", orig_model, decision, original, result)
        else:
            body_result = await self._reformulate(body_text, orig_model)
            if body_result is not None:
                result = marker + body_result
                self._cache_put(original, result)
            else:
                result = None

        if result is None:
            self.counters.failed += 1
            log.info("[model=%s] reformulation echec -> envoi original: %.80r", orig_model, original)
            return await self.relay(request, data=raw)

        last["content"] = result
        self.counters.reformulated += 1
        elapsed = time.monotonic() - started
        log.info(
            "[model=%s] %s: %.80r -> %.80r (%.1fs)",
            orig_model, decision, original, result, elapsed,
        )

        # La note "Reformulé" n'est injectee que sur une reformulation FRAICHE :
        # un cache hit = le meme message a deja ete reformule recemment (tours de
        # boucle agentique de Hermes) -> le contenu est remplace silencieusement,
        # sans re-poller le chat avec la note a chaque reponse.
        notice = ""
        if cached is None and self.cfg["show_rewrite"]:
            notice = f"> **Reformulé** : \"{result}\"\n\n"

        return await self._forward(body, request, notice)

    def _cache_get(self, message: str) -> str | None:
        ttl = self.cfg.get("cache_ttl", 0)
        if ttl <= 0:
            return None
        entry = self.cache.get(message)
        if entry is None:
            return None
        ts, result = entry
        if time.monotonic() - ts > ttl:
            del self.cache[message]
            return None
        return result

    def _cache_put(self, message: str, result: str) -> None:
        ttl = self.cfg.get("cache_ttl", 0)
        if ttl <= 0:
            return
        if len(self.cache) >= 512:
            self.cache.clear()
        self.cache[message] = (time.monotonic(), result)

    async def _reformulate(self, message: str, orig_model: str) -> str | None:
        combo_model = self.cfg["combo_model"]
        if "{model}" in combo_model:
            combo_model = combo_model.replace("{model}", orig_model or "default")

        if reformulate.decide(
            message, self.cfg["min_words"], self.cfg["skip_prefixes"],
        ) == "protect":
            self.counters.protected += 1
            text, blocks = reformulate.extract_blocks(message)
            new_text = await reformulate.call_upstream(
                text, combo_model, self.cfg["upstream"],
                self.cfg["api_key"], self.cfg["timeout"], self.session,
            )
            if new_text is None:
                return None
            return reformulate.restore_blocks(new_text, blocks)

        return await reformulate.call_upstream(
            message, combo_model, self.cfg["upstream"],
            self.cfg["api_key"], self.cfg["timeout"], self.session,
        )

    def _outbound_headers(self, request: web.Request) -> dict:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        if self.cfg["api_key"]:
            headers["Authorization"] = f"Bearer {self.cfg['api_key']}"
        elif "Authorization" not in headers:
            headers["Authorization"] = "Bearer "
        return headers

    async def _forward(self, body: dict, request: web.Request, notice: str = "") -> web.StreamResponse:
        url = self.upstream_url("/v1/chat/completions")
        headers = {"Content-Type": "application/json"}
        if self.cfg["api_key"]:
            headers["Authorization"] = f"Bearer {self.cfg['api_key']}"
        try:
            upstream_resp = await self.session.post(url, json=body, headers=headers)
        except Exception as exc:
            log.error("forward failed: %s", exc)
            return web.json_response({"error": {"message": f"watcher forward failed: {exc}"}}, status=502)

        if notice and body.get("stream"):
            resp = web.StreamResponse(status=upstream_resp.status)
            resp.headers["Content-Type"] = self._relay_content_type(upstream_resp)
            try:
                await resp.prepare(request)
                payload = json.dumps(
                    {"choices": [{"index": 0, "delta": {"content": notice},
                                  "finish_reason": None}]},
                    ensure_ascii=False,
                )
                await resp.write(f"data: {payload}\n\n".encode())
                async for chunk in upstream_resp.content.iter_any():
                    await resp.write(chunk)
                return resp
            except (ConnectionResetError, aiohttp.ClientConnectionResetError,
                    asyncio.CancelledError):
                log.debug("client disconnected during notice stream")
                return resp
            finally:
                await upstream_resp.release()

        if notice:
            raw = await upstream_resp.read()
            try:
                data = json.loads(raw)
                choice = data["choices"][0]["message"]
                if isinstance(choice.get("content"), str):
                    choice["content"] = notice + choice["content"]
                raw = json.dumps(data, ensure_ascii=False).encode()
            except (ValueError, KeyError, TypeError, IndexError):
                pass
            await upstream_resp.release()
            return web.Response(
                body=raw, status=upstream_resp.status,
                headers={"Content-Type": self._relay_content_type(upstream_resp)},
            )

        return await self._relay_response(upstream_resp, request)

    def _relay_content_type(self, upstream_resp) -> str:
        content_type = upstream_resp.headers.get("Content-Type", "application/json")
        if "charset=" not in content_type.lower():
            content_type += "; charset=utf-8"
        return content_type

    async def _relay_response(self, upstream_resp, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=upstream_resp.status)
        resp.headers["Content-Type"] = self._relay_content_type(upstream_resp)
        for header in ("Cache-Control", "X-Accel-Buffering", "x-request-id"):
            if upstream_resp.headers.get(header):
                resp.headers[header] = upstream_resp.headers[header]

        try:
            await resp.prepare(request)
            async for chunk in upstream_resp.content.iter_any():
                await resp.write(chunk)
            return resp
        except (ConnectionResetError, aiohttp.ClientConnectionResetError,
                asyncio.CancelledError):
            log.debug("client disconnected mid-relay (%s)", request.path_qs)
            return resp
        finally:
            await upstream_resp.release()


def main() -> None:
    cfg = build_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    log.info(
        "watcher demarre: listen=%s upstream=%s combo_model=%r min_words=%s",
        cfg["listen"], cfg["upstream"], cfg["combo_model"], cfg["min_words"],
    )

    app = WatcherApp(cfg)
    routes = web.Application(client_max_size=64 * 1024 * 1024)
    routes.router.add_get("/health", app.health)
    routes.router.add_post("/v1/chat/completions", app.chat_completions)
    routes.router.add_route("*", "/{tail:.*}", app.relay)
    routes.on_startup.append(lambda _: app.start())
    routes.on_cleanup.append(lambda _: app.stop())

    host, _, port = cfg["listen"].rpartition(":")
    web.run_app(routes, host=host, port=int(port), print=None)


if __name__ == "__main__":
    main()
