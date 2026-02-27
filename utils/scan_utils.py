import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from .file_utils import dir_check, safe_open
from .parser_utils import get_parser
from .sample_utils import Sample, find_parent_fn_node, node_of_unsafe_func, unsafe_block_query
from .batch_utils import PromptProvider, extract_sample_result

if TYPE_CHECKING:
    from .LLM_utils import ChatModel


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "target",
    "node_modules",
}

DEFAULT_EXCLUDE_DIRS_NO_TESTS = DEFAULT_EXCLUDE_DIRS.union({"tests", "benches", "examples"})


def iter_rust_files(crate_root: str, include_tests: bool = False) -> Iterable[str]:
    exclude_dirs = DEFAULT_EXCLUDE_DIRS if include_tests else DEFAULT_EXCLUDE_DIRS_NO_TESTS
    for root, dirs, files in os.walk(crate_root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".rs"):
                yield os.path.join(root, file)


def extract_candidates(crate_root: str, include_tests: bool = False, limit: Optional[int] = None) -> List[dict]:
    parser = get_parser()
    candidates: Dict[Tuple[str, int], dict] = {}
    crate_root = os.path.abspath(crate_root)
    for abs_file in iter_rust_files(crate_root, include_tests=include_tests):
        try:
            with open(abs_file, "r", encoding="utf-8") as f:
                code = f.read()
        except OSError:
            continue
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        captures = unsafe_block_query.captures(tree.root_node)
        for node, _ in captures:
            fn_node = find_parent_fn_node(node)
            if fn_node is None:
                continue
            if node_of_unsafe_func(fn_node):
                continue
            name_node = fn_node.child_by_field_name("name")
            if name_node is None:
                continue
            fn_name = code_bytes[name_node.start_byte() : name_node.end_byte()].decode("utf-8", errors="ignore")
            start_line = fn_node.start_point[0]
            key = (abs_file, start_line)
            if key in candidates:
                continue
            rel_file = os.path.relpath(abs_file, crate_root)
            candidates[key] = {
                "relative_file": rel_file,
                "start_line": start_line,
                "end_line": fn_node.end_point[0],
                "fn_name": fn_name,
            }
            if limit is not None and len(candidates) >= limit:
                return list(candidates.values())
    return list(candidates.values())


def upsert_toml_section(path: str, section: str, lines: List[str]) -> None:
    content = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    if content and not content.endswith("\n"):
        content += "\n"
    section_header = f"[{section}]"
    new_section = section_header + "\n" + "\n".join(lines) + "\n"
    pattern = re.compile(rf"^\[{re.escape(section)}\]\s*$", re.M)
    match = pattern.search(content)
    if not match:
        content = content + ("\n" if content and not content.endswith("\n\n") else "") + new_section
    else:
        next_match = re.compile(r"^\[.+?\]\s*$", re.M).search(content, match.end())
        end = next_match.start() if next_match else len(content)
        content = content[: match.start()] + new_section + content[end:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def run_context_retriever(
    repo_root_path: str,
    repo_name: str,
    item_root_path: str,
    result_root_path: str,
    target: str = "scan",
) -> str:
    context_dir = os.path.join(os.path.dirname(__file__), "..", "context_retriever")
    context_dir = os.path.abspath(context_dir)
    config_path = os.path.join(context_dir, "config.toml")
    upsert_toml_section(
        config_path,
        target,
        [
            'function_locate = "start_line"',
            f'item_root_path = "{item_root_path}"',
            f'repo_root_path = "{repo_root_path}"',
            f'result_root_path = "{result_root_path}"',
        ],
    )
    cmd = ["cargo", "run", "--manifest-path", "Cargo.toml", target]
    subprocess.run(cmd, cwd=context_dir, check=True)
    return os.path.join(result_root_path, f"{repo_name}.json")


def make_decomposer(chatbot: "ChatModel", prompt_name: str) -> Any:
    from decompose_safety_and_classify import Decomposer

    return Decomposer(chatbot, prompt_name)


def add_constraints(
    samples: List[dict],
    decomposer: Any,
) -> List[dict]:
    from decompose_safety_and_classify import fine_grained_questions_of_sample

    for sample in samples:
        sample["constraints"] = fine_grained_questions_of_sample(decomposer, sample)
    return samples


@dataclass
class EvaluationResult:
    sample_label: str
    result: str
    response: List[dict]


class InteractiveEvaluator:
    def __init__(self, prompt_info: dict, chatbot: "ChatModel"):
        self.prompt_info = prompt_info
        self.chatbot = chatbot
        self.prompt_provider = PromptProvider(prompt_info)
        self.fine_grained_check = prompt_info.get("fine_grained_check", True)
        example_enabled = prompt_info.get("example_strategy", {}).get("enabled", False)
        num_yes_examples = prompt_info.get("example_strategy", {}).get("num_yes_examples", "")
        self.one_pattern_a_time = example_enabled and num_yes_examples == "one_pattern_a_time"
        self.temperature = prompt_info.get("temperature", 0.0)
        self.seed = prompt_info.get("seed", 2024)

    def evaluate_sample(self, sample_raw: dict) -> EvaluationResult:
        sample = Sample(sample_raw)
        result = "sound"
        responses: List[dict] = []
        if self.fine_grained_check:
            for fine_grained_q in sample_raw.get("constraints", []):
                if self.one_pattern_a_time:
                    sub_responses = []
                    prompts = self.prompt_provider.get_prompts_for_one_pattern_a_time(sample, fine_grained_q)
                    q_result = "No"
                    for messages in prompts:
                        content = self.chatbot.send_messages(
                            messages, temperature=self.temperature, seed=self.seed
                        )
                        pred = extract_sample_result(content)
                        sub_responses.append({"result": pred, "content": content})
                        if pred == "Yes":
                            q_result = "Yes"
                        elif pred == "unknown" and q_result == "No":
                            q_result = "unknown"
                    responses.append(
                        {"question": fine_grained_q, "result": q_result, "sub_response": sub_responses}
                    )
                else:
                    messages = self.prompt_provider.get_prompt(sample, fine_grained_q)
                    content = self.chatbot.send_messages(messages, temperature=self.temperature, seed=self.seed)
                    pred = extract_sample_result(content)
                    responses.append({"question": fine_grained_q, "result": pred, "response": content})

                q_result = responses[-1]["result"]
                if q_result == "No":
                    result = "unsound"
                elif result == "sound" and q_result == "unknown":
                    result = "unknown"
        else:
            for unsafe_callee in sample.unsafe_callees:
                if len(unsafe_callee.get("safety", "")) == 0:
                    continue
                messages = self.prompt_provider.get_prompt(
                    sample,
                    {
                        "content": unsafe_callee["safety"],
                        "fn_name": unsafe_callee["name"],
                        "type": "unknown",
                    },
                )
                content = self.chatbot.send_messages(messages, temperature=self.temperature, seed=self.seed)
                pred = extract_sample_result(content)
                responses.append({"question": unsafe_callee["safety"], "result": pred, "response": content})
                if pred == "No":
                    result = "unsound"
                elif result == "sound" and pred == "unknown":
                    result = "unknown"

        return EvaluationResult(sample_label=sample.sample_label, result=result, response=responses)

    def evaluate_samples(self, samples: List[dict]) -> List[dict]:
        results = []
        for sample in samples:
            if len(sample.get("constraints", [])) == 0 and self.fine_grained_check:
                continue
            results.append(self.evaluate_sample(sample).__dict__)
        return results
