"""Thin client for the local Ollama server (http://127.0.0.1:11434).

Ollama is a free program that runs open-source LLMs on your own computer and
exposes a small HTTP API. We talk to it with plain `requests` calls, so no
extra Python package is needed and nothing ever leaves your machine.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import requests

from .utils import LectureMateError

DEFAULT_MODEL = os.environ.get("LECTUREMATE_MODEL", "qwen2.5:7b")
NUM_CTX = 8192  # context window we ask Ollama for (tokens)


def _normalise_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith("http"):
        host = f"http://{host}"
    return host


DEFAULT_HOST = _normalise_host(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))

NOT_RUNNING_MSG = (
    "Cannot reach Ollama. Start the Ollama app (or run `ollama serve` in a terminal) "
    "and try again."
)


class LLMError(LectureMateError):
    """Raised when the local LLM cannot be reached or fails."""


class OllamaClient:
    def __init__(self, model: str | None = None, host: str | None = None, num_ctx: int = NUM_CTX):
        self.model = (model or DEFAULT_MODEL).strip()
        self.host = _normalise_host(host) if host else DEFAULT_HOST
        self.num_ctx = num_ctx

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        """Quick health check used by the sidebar."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
        except requests.exceptions.RequestException:
            return {"running": False, "model_ready": False, "models": [], "message": NOT_RUNNING_MSG}

        wanted = self._full_model_name()
        ready = wanted in names
        message = "" if ready else (
            f"Model '{self.model}' is not downloaded. Run: ollama pull {self.model}"
        )
        return {"running": True, "model_ready": ready, "models": names, "message": message}

    def _full_model_name(self) -> str:
        return self.model if ":" in self.model else f"{self.model}:latest"

    # ------------------------------------------------------------------- chat
    def _payload(
        self,
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
        json_mode: bool,
        temperature: float,
        num_predict: int,
        stream: bool,
    ) -> dict[str, Any]:
        messages = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "10m",  # unload the model from RAM after 10 idle minutes
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    def _check_response(self, resp: requests.Response) -> None:
        if resp.status_code == 200:
            return
        try:
            detail = resp.json().get("error", "")
        except ValueError:
            detail = (resp.text or "")[:200]
        if resp.status_code == 404:
            raise LLMError(
                f"Model '{self.model}' was not found in Ollama. "
                f"Download it with: ollama pull {self.model}"
            )
        raise LLMError(f"Ollama returned an error ({resp.status_code}): {detail or 'unknown error'}")

    def chat(
        self,
        system: str,
        user: str,
        *,
        history: list[dict[str, str]] | None = None,
        json_mode: bool = False,
        temperature: float = 0.3,
        num_predict: int = 2048,
    ) -> str:
        """Send one request and return the full answer."""
        payload = self._payload(system, user, history, json_mode, temperature, num_predict, False)
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=(5, 900))
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(NOT_RUNNING_MSG) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(
                "The model took too long to answer (15 min). Try a smaller model, e.g. llama3.2:3b."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Request to Ollama failed: {exc}") from exc

        self._check_response(resp)
        try:
            content = resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama sent an unexpected response. Try again.") from exc
        content = (content or "").strip()
        if not content:
            raise LLMError("The model returned an empty answer. Try again.")
        return content

    def stream(
        self,
        system: str,
        user: str,
        *,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.3,
        num_predict: int = 2048,
    ) -> Iterator[str]:
        """Yield the answer piece by piece (for the chat UI)."""
        payload = self._payload(system, user, history, False, temperature, num_predict, True)
        try:
            with requests.post(
                f"{self.host}/api/chat", json=payload, stream=True, timeout=(5, 900)
            ) as resp:
                self._check_response(resp)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise LLMError(f"Ollama error: {data['error']}")
                    piece = (data.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(NOT_RUNNING_MSG) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError("The model took too long to answer. Try a smaller model.") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Request to Ollama failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("Ollama sent an unreadable response. Try again.") from exc
