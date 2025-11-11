"""
deepseek_module.py

Универсальный интерфейс DeepSeek API с автоматическим повтором, кешем и переводом на русский.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests


API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
CACHE_FILE = "output/deepseek_responses.jsonl"


# ---------------- Вспомогательные функции ---------------- #


def _ensure_cache_dir() -> None:
    cache_path = Path(CACHE_FILE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        cache_path.touch()


def _hash_key(task: str, text: str) -> str:
    return hashlib.md5((task + text).encode("utf-8")).hexdigest()


def _cached_get(key: str) -> Optional[str]:
    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("key") == key:
                    return payload.get("response")
    except OSError:
        return None
    return None


def _cached_set(key: str, resp: str) -> None:
    _ensure_cache_dir()
    with open(CACHE_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "response": resp}, ensure_ascii=False) + "\n")


def _load_api_key() -> Optional[str]:
    if API_KEY:
        return API_KEY
    env_path = Path(".env")
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if not line or line.strip().startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "DEEPSEEK_API_KEY":
                    return value.strip()
        except OSError:
            return None
    return None


def _call_deepseek(prompt: str, retries: int = 3, delay: int = 3) -> str:
    """Универсальный вызов DeepSeek Chat API с повтором и обработкой ошибок."""
    api_key = _load_api_key()
    if not api_key:
        return "unavailable"

    url = f"{API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a semantic analyzer for historical English texts.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=40)
            if resp.status_code == 200:
                result = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if result:
                    return result
            else:
                print(f"[DeepSeek {attempt + 1}] HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"[DeepSeek exception] {exc}")
        time.sleep(delay)
    return "unavailable"


def _cached_request(task: str, text: str, prompt: str) -> str:
    key = _hash_key(task, text or "")
    cached = _cached_get(key)
    if cached is not None:
        return cached
    result = _call_deepseek(prompt)
    _cached_set(key, result)
    return result


# ---------------- Основные функции ---------------- #


def classify_context(text: str) -> str:
    prompt = (
        "Classify this text about Kalmyks into one of "
        "['ethnographic', 'functional', 'evaluative', 'religious', 'imperial'].\n"
        f"Text: {text}"
    )
    return _cached_request("classify", text, prompt)


def detect_sentiment(text: str) -> str:
    prompt = (
        "Determine the overall attitude toward Kalmyks (positive, neutral, negative, or ambivalent).\n"
        f"Text: {text}"
    )
    return _cached_request("sentiment", text, prompt)


def summarize_context(text: str) -> str:
    prompt = (
        "Provide a short, factual summary (1–2 sentences) of this text about Kalmyks.\n"
        f"Text: {text}"
    )
    return _cached_request("summary", text, prompt)


def translate_to_russian(text: str) -> str:
    """Перевод любого английского текста на русский (через DeepSeek)."""
    if not text or str(text).strip().lower() in {"unavailable", "none"}:
        return "нет данных"
    prompt = f"Translate this text into Russian, preserving scientific tone:\n{text}"
    return _cached_request("translate", text, prompt)


def interpret_table(title: str, sample: str) -> str:
    prompt = (
        f"Summarize the key trends in the table '{title}' "
        "in 4–5 sentences, using a scholarly and analytical tone. "
        "Do not restate the data. Identify what the results mean in cultural, historical, "
        "or linguistic terms. Write concisely, as for an academic report. "
        f"Table preview:\n{sample}"
    )
    return _cached_request("interpret-table", f"{title}:{sample}", prompt)


def interpret_cached(title: str, description_hint: str, sample_text: str = "") -> str:
    cache_payload = f"{title}|{description_hint}|{sample_text[:800]}"
    prompt = (
        "You are an academic analyst working on a Digital Humanities project "
        "about representations of Kalmyks in 19th–20th century English travelogues.\n\n"
        f"Task: Provide a concise scholarly interpretation (5–6 sentences) of the visualization titled '{title}'.\n"
        "Explain what trends or cultural implications the data show, how they should be read, "
        "and what they reveal about British views of the Kalmyks or Siberia.\n"
        "Respond in Russian, adopting a formal academic tone that could appear in a research article.\n\n"
        f"Contextual description: {description_hint}\n\n"
        f"Sample data preview:\n{sample_text[:800]}"
    )
    return _cached_request("interpret-visual", cache_payload, prompt)


def request_commentary(prompt: str, task: str = "commentary") -> str:
    return _cached_request(task, prompt, prompt)


__all__ = [
    "classify_context",
    "detect_sentiment",
    "summarize_context",
    "translate_to_russian",
    "interpret_table",
    "request_commentary",
    "interpret_cached",
]

