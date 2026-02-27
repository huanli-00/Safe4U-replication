import openai
import json
import tiktoken
from .file_utils import safe_open
import os
from typing import List
import numpy as np
from scipy import spatial
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from jinja2.exceptions import TemplateError
import logging

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
    def __init__(self, model: str, temperature=0, seed=2024):
        def is_int(s: str):
            try:
                int(s)
                return True
            except ValueError:
                return False

        self.temperature = temperature
        self.seed = seed
        if model.startswith("gpt-"):
            self.openai = True
            self.model = model
            self.model_name = model
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.max_length = MAX_TOKEN
        elif is_int(model):
            self.openai = True
            # Deployed OpenAI-compatible with vLLM
            # vllm_port = os.environ.get("VLLM_PORT")
            vllm_port = int(model)
            self.client = openai.OpenAI(base_url=f"http://127.0.0.1:{vllm_port}/v1")
            model_info = self.client.models.list().to_dict(mode="json")
            self.model = model_info["data"][0]["id"]
            self.model_name = self.model.split("/")[-1]
            self.max_length = model_info["data"][0]["max_model_len"]
            logging.info(f"The Model of {vllm_port} is {self.model}")
        else:
            self.openai = False

            self.llm = LLM(model=model)
            self.model = model
            self.model_name = model.split("/")[-1]
            self.sampling_params = SamplingParams(temperature=temperature, seed=seed, max_tokens=MAX_RESPONSE_TOKEN)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.max_length = self.tokenizer.model_max_length

    def send_messages(self, messages, temperature=None, seed=None, max_tokens: int = None):
        if self.openai:
            messages = check_num_token(messages, self.max_length)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                seed=seed or self.seed,
                max_tokens=max_tokens or MAX_RESPONSE_TOKEN,
            )
            return response.choices[0].message.content
        else:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = self.llm.generate([prompt], self.sampling_params)
            return response[0].outputs[0].text


class EmbeddingModel:
    def __init__(self, model="text-embedding-3-small"):
        embd_url = os.environ.get("EMBEDDING_URL")
        if embd_url is None:
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_KEY"))
            self.model = model
        else:
            # setup other OPENAI-compatible embedding model
            self.client = openai.OpenAI(base_url=embd_url, api_key=os.environ.get("EMBEDDING_KEY"))
            self.model = model

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


class LocalBatch:
    def __init__(self, llm, batch_file: str, result_file: str) -> None:
        batch_data_lines = open(batch_file).readlines()
        json_objs = [json.loads(line) for line in batch_data_lines]

        obj = json_objs[0]
        temperature = obj["body"]["temperature"]
        seed = obj["body"]["seed"]
        model = obj["body"]["model"]
        tokenizer = AutoTokenizer.from_pretrained(model)
        print("The max_length of model is: ", tokenizer.model_max_length)

        prompts = list()
        # ATTENTION: the chat_template in huggingface transformers may not work for all models, ans some template may be error, please check the template before using it.
        for obj in json_objs:
            messages = obj["body"]["messages"]
            messages = check_num_token(messages, tokenizer.model_max_length)
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except TemplateError:
                # Get template error because the model does not support 'system' role
                # So replace the role of the first message to 'user' and add a fake answer to the second message
                # Reserve some space for the new message
                messages = check_num_token(messages, tokenizer.model_max_length - 64)
                messages[0]["role"] = "user"
                messages.insert(1, {"role": "assistant", "content": "Okay, I understand."})
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        sampling_params = SamplingParams(temperature=temperature, seed=seed, max_tokens=MAX_RESPONSE_TOKEN)

        outputs = llm.generate(prompts, sampling_params)

        result_writer = open(result_file, "w")
        for idx, output in enumerate(outputs):
            prompt = output.prompt
            response = output.outputs[0]
            # Use the same response template as OpenAI Batch
            # so that the result can be resolved by the same function
            wrapped_res = {
                "id": output.request_id,
                "custom_id": json_objs[idx]["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response.text,
                                },
                            }
                        ],
                    },
                },
            }
            result_writer.write(json.dumps(wrapped_res))
            result_writer.write("\n")
        result_writer.close()


class BatchUtil:
    def __init__(self, model: str = "gpt-3.5-turbo", task: str = "completions"):
        self.task = task
        self.model = model
        self.samples = list()
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.openai = self.model.startswith("gpt-")

    def add_to_batch(self, custom_id: str, messages, temperature=0.0, seed=2024):
        self.samples.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": f"/v1/chat/{self.task}",
                "body": {
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": MAX_RESPONSE_TOKEN,
                    "seed": seed,
                    "messages": messages,
                },
            }
        )

    def save_batch(
        self,
        batch_file: str,
    ):
        with safe_open(batch_file, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")
        self.samples = list()

    def upload_and_create_batch(self, file_path: str):
        if not self.openai:
            raise NotImplementedError("Local LLM is not supported yet, use process_batch instead.")
        batch_input_file = self.client.files.create(file=open(file_path, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        metadata = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "nightly eval job"},
        )
        return metadata

    def process_batch(self, llm, batch_file: str, result_file):
        LocalBatch(llm, batch_file, result_file)

    def save_and_upload_batch(self, batch_file: str):
        self.save_batch()
        return self.upload_and_create_batch(batch_file)

    def retrieve_batch_result(self, batch_id: str, output_file: str):
        if not self.openai:
            print("Local LLM has no batch job to retrieve.")
            pass
        status = self.client.batches.retrieve(batch_id)
        if status.status == "completed":
            batch_output_file_id = status.output_file_id
            content = self.client.files.content(batch_output_file_id)
            with open(output_file, "wb") as f:
                f.write(content.content)
        else:
            print(status)
            
            print("The batch job is not completed yet.")
