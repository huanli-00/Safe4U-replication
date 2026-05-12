import openai
import tiktoken
from typing import List
import numpy as np
from scipy import spatial
import logging

from .config_utils import chat_extra_body, load_env_config

MAX_RESPONSE_TOKEN = 1536
MAX_TOKEN = 16384


def num_tokens_from_messages(messages):
    """
    Calculate the number of tokens in the messages.
    Reference:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )  # model to encoding mapping https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def num_tokens_str(message):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(message))


def check_num_token(messages, max_length=MAX_TOKEN):
    conv_history_tokens = num_tokens_from_messages(messages)
    while conv_history_tokens + MAX_RESPONSE_TOKEN >= max_length:
        # delete the first QA pair
        del messages[2]
        del messages[1]
        conv_history_tokens = num_tokens_from_messages(messages)
    return messages


def add_to_message(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages


class ChatModel:
    def __init__(self, model: str = "", temperature=0, seed=2024):
        def is_int(s: str):
            try:
                int(s)
                return True
            except ValueError:
                return False

        self.temperature = temperature
        self.seed = seed
        config = load_env_config()
        model = model or str(config.get("model", "")).strip()
        base_url = str(config.get("base_url", "")).strip()
        api_key = str(config.get("api_key", "")).strip()
        timeout = float(config.get("request_timeout", 120))
        max_tokens = int(config.get("max_tokens", 4096))
        self.extra_body = chat_extra_body(config, model)
        if base_url:
            self.openai = True
            self.model = model
            self.model_name = model.split("/")[-1]
            self.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key or "EMPTY",
                timeout=timeout,
                max_retries=0,
            )
            self.max_length = MAX_TOKEN
            self.max_tokens = max_tokens
        elif model.startswith("gpt-"):
            self.openai = True
            self.model = model
            self.model_name = model
            self.client = openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
            self.max_length = MAX_TOKEN
            self.max_tokens = max_tokens
        elif is_int(model):
            self.openai = True
            local_port = int(model)
            self.client = openai.OpenAI(
                base_url=f"http://127.0.0.1:{local_port}/v1",
                api_key=api_key or "EMPTY",
                timeout=timeout,
                max_retries=0,
            )
            model_info = self.client.models.list().to_dict(mode="json")
            self.model = model_info["data"][0]["id"]
            self.model_name = self.model.split("/")[-1]
            self.max_length = model_info["data"][0]["max_model_len"]
            self.max_tokens = max_tokens
            logging.info(f"The model served on port {local_port} is {self.model}")
        else:
            raise ValueError(
                "Direct-use Safe4U expects an OpenAI model name or an OpenAI-compatible local port. "
                "For local models, serve them behind an OpenAI-compatible endpoint and set base_url in env.json."
            )

    def send_messages(self, messages, temperature=None, seed=None, max_tokens: int = None):
        if self.openai:
            messages = check_num_token(messages, self.max_length)
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "seed": seed or self.seed,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            }
            if self.extra_body:
                request_kwargs["extra_body"] = self.extra_body
            response = self.client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message
            return message.content or getattr(message, "reasoning_content", "") or ""

class EmbeddingModel:
    def __init__(self, model: str = ""):
        config = load_env_config()
        self.model = model or str(config.get("embedding_model", "")).strip() or "text-embedding-3-small"
        embedding_url = str(config.get("embedding_url", "")).strip()
        embedding_key = str(config.get("embedding_key", "")).strip()
        api_key = str(config.get("api_key", "")).strip()
        timeout = float(config.get("request_timeout", 120))
        if embedding_url:
            self.client = openai.OpenAI(
                base_url=embedding_url,
                api_key=embedding_key or "EMPTY",
                timeout=timeout,
                max_retries=0,
            )
        else:
            self.client = openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        # replace newlines, which can negatively affect performance.
        list_of_text = [text.replace("\n", " ") for text in list_of_text]

        data = self.client.embeddings.create(input=list_of_text, model=self.model).data
        return [d.embedding for d in data]


def k_nearest_indices(
    embedding,
    embeddings: List[List[float]],
    k=1,
    distance_metric="cosine",
):
    """
    Return the top k nearest sample by computing distance based on embeddings.
    """
    # OpenAI embeddings are normalized to length 1, which means that:
    # - Cosine similarity can be computed slightly faster using just a dot product
    # - Cosine similarity and Euclidean distance will result in the identical rankings
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [distance_metrics[distance_metric](embedding, emb) for emb in embeddings]
    # The indices of the k smallest distances which means k most similar samples.
    indices = np.argsort(distances)[:k]
    return indices
