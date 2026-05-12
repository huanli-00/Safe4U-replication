import json
import os
from typing import Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, "env.json")

DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "base_url": "",
    "api_key": "",
    "embedding_model": "text-embedding-3-small",
    "embedding_url": "",
    "embedding_key": "",
    "request_timeout": 120,
    "max_tokens": 4096,
    "thinking": "",
    "extra_body": {},
}


def load_env_config(path: str = ENV_FILE) -> dict[str, Any]:
    config = DEFAULT_ENV_CONFIG.copy()
    if not os.path.exists(path):
        return config
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    for key, value in loaded.items():
        if key in config and value is not None:
            config[key] = value
    return config


def apply_prompt_config(prompt_info: dict, config: dict[str, Any]) -> None:
    embedding_model = str(config.get("embedding_model", "")).strip()
    if not embedding_model:
        return
    example_strategy = prompt_info.get("example_strategy")
    if isinstance(example_strategy, dict) and "embedding_model" in example_strategy:
        example_strategy["embedding_model"] = embedding_model


def chat_extra_body(config: dict[str, Any], model: str) -> dict[str, Any]:
    extra_body = config.get("extra_body", {})
    if not isinstance(extra_body, dict):
        extra_body = {}
    extra_body = extra_body.copy()
    thinking = str(config.get("thinking", "")).strip().lower()
    if not thinking and model.startswith("deepseek-v4"):
        thinking = "disabled"
    if thinking:
        extra_body["thinking"] = {"type": thinking}
    return extra_body
