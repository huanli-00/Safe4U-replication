import json
import os
import utils.batch_utils as bu
from utils.LLM_utils import BatchUtil
from utils.file_utils import dir_check, copy_to
import argparse
from vllm import LLM

prompt_dir = "prompt/"
sample_dir = "data/samples/"
result_root_dir = f"result/"


class BatchAnalyzer:
    def __init__(self, prompt: str, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.model_name = model.split("/")[-1]
        self.result_dir = os.path.join(result_root_dir, self.model_name, prompt)
        prompt_file = os.path.join(prompt_dir, f"{prompt}.json")
        self.prompt_info = json.load(open(prompt_file, "r"))
        dir_check(self.result_dir)
        copy_to(prompt_file, self.result_dir)
        self.split_safety_prompt = self.prompt_info["split_safety_prompt"]
        self.batch_id_dict_file = os.path.join(self.result_dir, f"batch_id.json")
        self.batch_generator = bu.BatchGenerator(self.prompt_info, self.model)
        self.result_resolver = bu.ResultResolver(self.prompt_info)
        self.batch_util = BatchUtil(self.model)

    def sample_file(self, sample_target: str) -> str:
        if self.split_safety_prompt == "none":
            return os.path.join(sample_dir, "original", f"{sample_target}.json")
        return os.path.join(sample_dir, self.model_name, self.split_safety_prompt, f"{sample_target}_fine_grained.json")

    def batch_file(self, sample_target: str) -> str:
        return os.path.join(self.result_dir, f"{sample_target}_batch.jsonl")

    def batch_result_file(self, sample_target: str) -> str:
        return os.path.join(self.result_dir, f"{sample_target}_batch_result.jsonl")

    def resolved_result_file(self, sample_target: str) -> str:
        return os.path.join(self.result_dir, f"{sample_target}_result.json")

    def generate_batch(self, sample_target: str):
        sample_file = self.sample_file(sample_target)
        batch_file = self.batch_file(sample_target)
        self.batch_generator.generate_batch_file(sample_file, batch_file)

    def process_batch(self, llm, sample_target: str):
        batch_file = self.batch_file(sample_target)
        result_file = self.batch_result_file(sample_target)
        self.batch_util.process_batch(llm, batch_file, result_file)

    def resolve_batch_result(self, sample_target: str):
        batch_result_file = self.batch_result_file(sample_target)
        resolved_result_file = self.resolved_result_file(sample_target)
        self.result_resolver.resolve_batch_result(
            batch_result_file, resolved_result_file
        )

    def download_and_resolve_batch_result(self, sample_target: str):
        self.download_batch_result(sample_target)
        self.resolve_batch_result(sample_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Analyzer")
    parser.add_argument(
        "--prompt", nargs="*", type=str, help="Prompt name", default=["Safe4U"]
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name(OpenAI) or path(Local Models)",
        required=True,
    )
    parser.add_argument(
        "--target",
        nargs="+",
        type=str,
        help="Sample target (risky, filtered_unsafe, 11cve, scan)",
        default="risky",
        required=True,
    )
    parser.add_argument("--device", type=int, help="CUDA device number", default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    llm = LLM(model=args.model)
    for prompt in args.prompt:
        for target in args.target:
            print(f"Prompt: {prompt}, Target: {target}, Device: {args.device}")
            analyzer = BatchAnalyzer(prompt, args.model)
            analyzer.generate_batch(target)
            analyzer.process_batch(llm, target)
            analyzer.resolve_batch_result(target)
