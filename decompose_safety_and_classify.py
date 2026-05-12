import logging

logging.basicConfig(format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO)
# logging.basicConfig(format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import json
import os
import utils.LLM_utils as llm
import utils.sample_utils as su
import utils.parser_utils as pu
from utils.file_utils import safe_open, dir_check

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
split_dir = os.path.join(PROJECT_ROOT, "result", "split_safety")
dir_check(split_dir)
prompt_dir = os.path.join(PROJECT_ROOT, "prompt")
example_dir = os.path.join(PROJECT_ROOT, "examples")


def construct_fn_context(unsafe_fn: dict, self_info: dict = None) -> str:
    context = ""
    if self_info is not None and len(self_info) > 0:
        self_context = su.add_prefix_to(
            f"Self Description",
            su.wrap_in_text(su.get_description_from_doc(self_info["documentation"])),
        )
        self_context += su.add_prefix_to("Self Declaration", su.wrap_in_code_block(self_info["declaration"], "rust"))
        context += su.add_section_title("### Self Information", self_context)
    context += su.add_section_title(
        "### Function Declaration",
        su.wrap_in_code_block(unsafe_fn["declaration"], "rust"),
    )
    context += su.add_section_title("### Description", su.wrap_in_text(unsafe_fn["description"]))
    context += su.add_section_title("### Safety Section", su.wrap_in_text(unsafe_fn["safety"]))
    context = su.add_section_title("## Function Context", context)
    return context


class Refiner:
    def __init__(self, chatbot: llm.ChatModel, critique_prompt: str, refine_prompt: str, round_limit: int):
        self.chatbot = chatbot
        self.critique_prompt = json.load(open(os.path.join(prompt_dir, f"{critique_prompt}.json"), "r"))
        self.refine_prompt = json.load(open(os.path.join(prompt_dir, f"{refine_prompt}.json"), "r"))
        self.round_limit = round_limit
        self.critique_system = "\n".join(self.critique_prompt["system"]["content"])
        self.critique_template = "\n".join(self.critique_prompt["question"])
        self.critique_temperature = self.critique_prompt.get("temperature", 0)
        self.critique_seed = self.critique_prompt.get("seed", 2024)
        self.refine_system = "\n".join(self.refine_prompt["system"]["content"])
        self.refine_template = "\n".join(self.refine_prompt["question"])
        self.refine_temperature = self.refine_prompt.get("temperature", 0)
        self.refine_seed = self.refine_prompt.get("seed", 2024)

    def critique_and_refine(self, decomposed_contracts: str, fn_context: str):
        round_cnt = 0
        log = []
        while round_cnt < self.round_limit:
            # generate critique
            messages = []
            llm.add_to_message(messages, "system", self.critique_system)
            llm.add_to_message(
                messages, "user", self._fill_template(self.critique_template, fn_context, decomposed_contracts, "")
            )
            critique = self.chatbot.send_messages(messages, temperature=self.critique_temperature, seed=self.critique_seed)
            logging.debug(f"Critique: {critique}")
            log.append({"role": "critique", "content": critique})
            critique_result = self._extract_critique_result(critique)
            if critique_result == "Yes":
                return decomposed_contracts, log
            # refine
            messages = []
            llm.add_to_message(messages, "system", self.refine_system)
            llm.add_to_message(
                messages, "user", self._fill_template(self.refine_template, fn_context, decomposed_contracts, "")
            )
            decomposed_contracts = self.chatbot.send_messages(
                messages, temperature=self.refine_temperature, seed=self.refine_seed
            )
            logging.debug(f"Refined contracts: {decomposed_contracts}")
            log.append({"role": "refine", "content": decomposed_contracts})
            round_cnt += 1
        return decomposed_contracts, log

    def _fill_template(self, template: str, fn_context: str, decomposed_contracts: str, critique: str):
        filled_template = template.replace("%{FUNCTION_CONTEXT}", fn_context)
        filled_template = filled_template.replace("%{DECOMPOSED_CONTRACTS}", decomposed_contracts)
        filled_template = filled_template.replace("%{CRITIQUE}", critique)
        return filled_template

    @staticmethod
    def _extract_critique_result(response: str) -> str:
        """
        Extract the critique result from the end of response.
        """
        lines = response.split("\n")
        for line in reversed(lines):
            if "**No**" not in line and "**Yes**" in line:
                return "Yes"
            elif "**Yes**" not in line and "**No**" in line:
                return "No"
            elif "**Yes**" in line and "**No**" in line:
                return "unknown"
        return "unknown"


class Classifier:
    def __init__(self, chatbot: llm.ChatModel, classification_prompt: str):
        self.chatbot = chatbot
        self.prompt_info = json.load(open(os.path.join(prompt_dir, f"{classification_prompt}.json"), "r"))
        self.system = "\n".join(self.prompt_info["system"]["content"])
        self.question_template = "\n".join(self.prompt_info["question"])
        self.temperature = self.prompt_info.get("temperature", 0)
        self.seed = self.prompt_info.get("seed", 2024)
        self.messages = []
        example_src = self.prompt_info.get("examples", "")
        llm.add_to_message(self.messages, "system", self.system)
        if len(example_src) > 0:
            example_file = os.path.join(example_dir, f"{example_src}.json")
            examples = json.load(open(example_file, "r"))
            for example in examples:
                query = self._fill_in_unsafe_fn_into_template(self.question_template, example)
                llm.add_to_message(self.messages, "user", query)
                llm.add_to_message(self.messages, "assistant", example["response_with_type"])

    def classify(self, unsafe_fn: dict, self_info: dict = None):
        messages = self.messages.copy()
        query = self._fill_in_unsafe_fn_into_template(self.question_template, unsafe_fn, self_info)
        llm.add_to_message(messages, "user", query)
        classification = self.chatbot.send_messages(messages, temperature=self.temperature, seed=self.seed)
        logging.debug(f"Classification: {classification}")
        return classification

    @staticmethod
    def _fill_in_unsafe_fn_into_template(template: str, unsafe_fn: dict, self_info: dict = None) -> str:
        """
        fill in unsafe function info into the template
        """
        filled_template = template.replace("%{UNSAFE_FUNCTION_TOKEN}", unsafe_fn["name"])
        filled_template = filled_template.replace("%{DECOMPOSED_CONTRACTS}", unsafe_fn["decomposed_contracts"])
        context = construct_fn_context(unsafe_fn, self_info)
        return filled_template.replace("%{FUNCTION_CONTEXT}", context)


class Decomposer:
    def __init__(self, chatbot: llm.ChatModel, prompt: str, chat_record: bool = True):
        self.chatbot = chatbot
        self.chat_record = chat_record
        prompt_file = os.path.join(prompt_dir, f"{prompt}.json")
        self.prompt = prompt
        self.prompt_info = json.load(open(prompt_file, "r"))
        self.temperature = self.prompt_info.get("temperature", 0)
        self.seed = self.prompt_info.get("seed", 2024)
        self.classification_strategy = self.prompt_info.get("classification_strategy", "simultaneous")
        if self.classification_strategy != "simultaneous":
            self.classifier = Classifier(chatbot, self.classification_strategy)
        system_prompt = "\n".join(self.prompt_info["system"]["content"])
        self.messages = []
        llm.add_to_message(self.messages, "system", system_prompt)
        self.question_template = "\n".join(self.prompt_info["question"])
        # * support few-shot learning
        example_src = self.prompt_info.get("examples", "")
        if len(example_src) > 0:
            example_file = os.path.join(example_dir, f"{example_src}.json")
            examples = json.load(open(example_file, "r"))
            response_key = "response_with_type" if self.classification_strategy == "simultaneous" else "decomposed_contracts"
            for example in examples:
                llm.add_to_message(
                    self.messages,
                    "user",
                    self._fill_in_unsafe_fn_into_template(self.question_template, example),
                )
                llm.add_to_message(self.messages, "assistant", example[response_key])
        self.self_check = self.prompt_info.get("critique", "") != "" and self.prompt_info.get("refine", "") != ""
        if self.self_check:
            critique_prompt = self.prompt_info.get("critique")
            refine_prompt = self.prompt_info.get("refine")
            round_limit = self.prompt_info.get("round_limit", 2)
            self.refiner = Refiner(chatbot, critique_prompt, refine_prompt, round_limit)

    def _history_fn_file(self, unsafe_fn_name: str) -> str:
        return os.path.join(split_dir, self.chatbot.model_name, self.prompt, f"{unsafe_fn_name}.json")

    def _decompose_with_llm(self, unsafe_fn: dict, self_info: dict = None):
        messages = self.messages.copy()
        question_with_context = self._fill_in_unsafe_fn_into_template(self.question_template, unsafe_fn, self_info)
        llm.add_to_message(messages, "user", question_with_context)
        decomposed_contracts = self.chatbot.send_messages(messages, temperature=self.temperature, seed=self.seed)
        logging.debug(f"Decomposed contracts: {decomposed_contracts}")
        log = [{"role": "decompose", "content": decomposed_contracts}]
        if self.self_check:
            fn_context = construct_fn_context(unsafe_fn, self_info)
            decomposed_contracts, refine_log = self.refiner.critique_and_refine(decomposed_contracts, fn_context)
            log.extend(refine_log)
        if self.classification_strategy != "simultaneous":
            unsafe_fn["decomposed_contracts"] = decomposed_contracts
            decomposed_contracts = self.classifier.classify(unsafe_fn, self_info)
            log.append({"role": "classification", "content": decomposed_contracts})
        return self._parse_result(decomposed_contracts, unsafe_fn), log

    def _parse_result(self, response: str, unsafe_fn: dict) -> list:
        # setting upper limit to prevent infinite loop output of LLMs.
        q_list = pu.parse_md_list(response)[:20]
        result = list()
        if len(q_list) == 0:
            result.append(
                {
                    "content": unsafe_fn["safety"],
                    "fn_name": unsafe_fn["name"],
                    "type": "Unknown",
                }
            )
        else:
            for q in q_list:
                # split the type and the content of contract.
                c_type, content = self._split_type_and_content_of_contract(q)
                result.append({"content": content, "fn_name": unsafe_fn["name"], "type": c_type})
        return result

    @staticmethod
    def _fill_in_unsafe_fn_into_template(template: str, unsafe_fn: dict, self_info: dict = None) -> str:
        """
        fill in unsafe function info into the template
        """
        filled_template = template.replace("%{UNSAFE_FUNCTION_TOKEN}", unsafe_fn["name"])
        context = construct_fn_context(unsafe_fn, self_info)
        return filled_template.replace("%{FUNCTION_CONTEXT}", context)

    @staticmethod
    def _split_type_and_content_of_contract(contract: str) -> tuple[str, str]:
        """
        split the type and the content of contract.
        """
        contract = contract.strip()
        # sep = ". ("
        # if sep in contract:
        #     content, c_type = contract.split(sep, 1)
        #     return c_type.strip(" \n)"), content.strip()
        sep = "("
        if sep in contract:
            snippets = contract.split(sep)
            c_type = snippets[-1]
            content = "(".join(snippets[:-1])
            return c_type.strip(" \n)"), content.strip()
        logging.warning(f"Failed to split type and content of contract: {contract}")
        return "Unknown", contract

    def decompose_contracts_of_unsafe_fn(
        self,
        unsafe_fn: dict,
        self_info: dict = None,
    ):
        """
        load split safety questions from history file if exists,
        otherwise call split_safety to generate questions.
        """
        # return empty list if no safety section
        if unsafe_fn["safety"] == "":
            return []
        history_fn_file = self._history_fn_file(unsafe_fn["name"])
        history_fn = list()
        if os.path.exists(history_fn_file):
            history_fn = json.load(open(history_fn_file, "r"))
            # find the matched one
            for fn in history_fn:
                if (
                    fn["declaration"] == unsafe_fn["declaration"]
                    and fn["src_mod"] == unsafe_fn["src_mod"]
                    and fn["safety"] == unsafe_fn["safety"]
                ):
                    return fn["questions"]

        split_result, log = self._decompose_with_llm(unsafe_fn, self_info)
        if len(split_result) == 0:
            logging.warning(f"Failed to split safety section of `{unsafe_fn['name']}`. Please manually handle it.")
        if self.chat_record:
            unsafe_fn["log"] = log
            unsafe_fn["round"] = len([res for res in log if res["role"] == "refine"])
        unsafe_fn["questions"] = split_result
        history_fn.append(unsafe_fn)
        json.dump(history_fn, safe_open(history_fn_file, "w"), indent=2)
        return unsafe_fn["questions"]


def fine_grained_questions_of_sample(decomposer: Decomposer, sample_raw: dict):
    """
    load split safety questions from history file if exists,
    otherwise call split_safety to generate questions.
    """
    sample_constraints = list()
    sample = su.Sample(sample_raw)
    for unsafe_fn in sample.unsafe_callees:
        questions = decomposer.decompose_contracts_of_unsafe_fn(unsafe_fn, sample.self_info)
        for q in questions:
            q["fn_name"] = unsafe_fn["name"]
        sample_constraints.extend(questions)
    return sample_constraints
