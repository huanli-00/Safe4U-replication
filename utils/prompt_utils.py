import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO)
import json
from .LLM_utils import *
import os
from .sample_utils import Sample, wrap_in_text
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
example_dir = os.path.join(PROJECT_ROOT, "examples")


def fill_question_slot(
    template: str,
    sample_detail: str,
    fine_grained_q: str,
    unsafe_fn_name: str,
    sample_fn_name: str,
    safety_section: str,
):
    question = template
    question = question.replace("%{DETAILED_SAMPLE}", sample_detail)
    question = question.replace("%{FINE_GRAINED_QUESTION}", fine_grained_q)
    question = question.replace("%{UNSAFE_FUNCTION_TOKEN}", unsafe_fn_name)
    question = question.replace("%{SAMPLE_FN_TOKEN}", sample_fn_name)
    question = question.replace("%{SAFETY_SECTION}", safety_section)
    return question


def extract_sample_result(response: str) -> str:
    """
    Extract the classification result from the end of response.
    """
    lines = response.split("\n")
    for last_line in reversed(lines[-5:]):
        if "**No**" not in last_line and "**Yes**" in last_line:
            return "Yes"
        elif "**Yes**" not in last_line and "**No**" in last_line:
            return "No"
        elif "**Yes**" in last_line and "**No**" in last_line:
            return "unknown"
        if "No" not in last_line and ("Yes" in last_line or "Yes\n" in response):
            return "Yes"
        elif "Yes" not in last_line and (
            "No" in last_line or "not sound" in last_line or "not correct" in last_line or "not meet" in last_line
        ):
            return "No"
        elif "Yes" in last_line and "No" in last_line:
            return "unknown"
    return "unknown"


class PromptProvider:
    def __init__(self, prompt_info: dict):
        self.prompt_info = prompt_info
        self.context_of_reference = prompt_info.get("context_of_reference", True)
        self.add_hints = prompt_info.get("add_hints", "none")
        self.fine_grained_check = prompt_info.get("fine_grained_check", True)
        prompt = list()
        if len(prompt_info["system"]["content"]) != 0:
            add_to_message(prompt, "system", "\n".join(prompt_info["system"]["content"]))
        self.prompt = prompt
        self.question_template = "\n".join(prompt_info["question"])

        # *load examples from prompt file
        example_strategy = prompt_info["example_strategy"]
        self.example_enabled = example_strategy.get("enabled", False)
        if self.example_enabled:
            self.source = example_strategy.get("source")
            self.criteria = example_strategy.get("criteria", "contract_type")
            if self.criteria == "contract_type":
                assert self.fine_grained_check, "Only fine_grained_contract (decomposed Safety) has contract_type"
            self.chain_of_thought = example_strategy.get("chain_of_thought", True)
            self.num_yes_examples = example_strategy.get("num_yes_examples", "one_pattern_a_time")
            if self.criteria == "random":
                assert isinstance(
                    self.num_yes_examples, int
                ), "Must specific the num_yes_examples for `random` example selection"
            self.num_no_examples = example_strategy.get("num_no_examples", 1)
            if self.criteria == "contract_type":
                assert (
                    self.num_yes_examples == "one_pattern_a_time" or self.num_yes_examples == "all_at_once"
                ), "Only support `one_pattern_a_time` and `all_at_once` for criteria == `contract_type`"
            if self.criteria in ["contract_type", "embedding"]:
                self.unknown_num_yes_examples = example_strategy["unknown_num_yes_examples"]
                self.unknown_selection_strategy = example_strategy["unknown_selection_strategy"]
                self.unknown_distance = example_strategy["unknown_distance"]
                self.embedding_model = EmbeddingModel(example_strategy["embedding_model"])
            self.__load_classified_examples()

    def __load_classified_examples(self):
        example_file = os.path.join(example_dir, f"{self.source}.json")
        original_samples = json.load(open(example_file, "r"))
        yes_examples = list()
        no_examples = list()
        yes_type_dict = dict()
        no_type_dict = dict()
        self.example_labels = list()
        for sample_raw in original_samples:
            sample = Sample(sample_raw)
            self.example_labels.append(sample_raw["sample_label"])
            for fine_grained_q in sample_raw["constraints"]:
                fine_grained_q["sample_label"] = sample_raw["sample_label"]
                # function to be checked
                fine_grained_q["sample_name"] = sample_raw["name"]
                # detailed question
                fine_grained_q["question"] = self.fill_question_slot(
                    self.get_sample_detail(sample),
                    fine_grained_q["content"],
                    fine_grained_q["fn_name"],
                    fine_grained_q["sample_name"],
                )
                # For chain-of-thought == false, rewrite the response
                if not self.chain_of_thought:
                    fine_grained_q["response"] = "## Answer:\n" + fine_grained_q["result"]
                if fine_grained_q["result"] == "Yes":
                    for t in fine_grained_q["type"]:
                        if t not in yes_type_dict:
                            yes_type_dict[t] = list()
                        yes_type_dict[t].append(len(yes_examples))
                    yes_examples.append(fine_grained_q)
                elif fine_grained_q["result"] == "No":
                    for t in fine_grained_q["type"]:
                        if t not in no_type_dict:
                            no_type_dict[t] = list()
                        no_type_dict[t].append(len(no_examples))
                    no_examples.append(fine_grained_q)
        self.yes_examples = yes_examples
        self.no_examples = no_examples
        if self.criteria == "contract_type" or self.criteria == "embedding":
            self.yes_embeddings = self.embedding_model.get_embeddings([example["content"] for example in yes_examples])
            self.no_embeddings = self.embedding_model.get_embeddings([example["content"] for example in no_examples])
            self.yes_type_dict = yes_type_dict
            self.no_type_dict = no_type_dict

    def get_sample_detail(self, sample: Sample):
        return sample.to_markdown(context_of_reference=self.context_of_reference, add_hints=self.add_hints)

    def contain_example(self, sample_label: str):
        return sample_label in self.example_labels

    def fill_question_slot(
        self,
        sample_detail: str,
        fine_grained_q: str,
        unsafe_fn_name: str,
        sample_fn_name: str,
    ):
        return fill_question_slot(
            self.question_template,
            sample_detail,
            wrap_in_text(fine_grained_q),
            unsafe_fn_name,
            sample_fn_name,
            "",
        )

    def __get_examples_idx_based_on_embedding(self, fine_grained_q: dict):
        if self.unknown_selection_strategy == "top_k":
            embedding = self.embedding_model.get_embeddings([fine_grained_q["content"]])[0]
            no_idx = k_nearest_indices(
                embedding,
                self.no_embeddings,
                self.num_no_examples,
                self.unknown_distance,
            )
            yes_idx = k_nearest_indices(
                embedding,
                self.yes_embeddings,
                self.unknown_num_yes_examples,
                self.unknown_distance,
            )
        else:
            logging.error("Unsupported strategy")
        return no_idx, yes_idx

    def __get_random_idx(self):
        no_idx = random.choices(range(len(self.no_examples)), k=self.num_no_examples)
        yes_idx = random.choices(range(len(self.yes_examples)), k=self.num_yes_examples)
        return no_idx, yes_idx

    def __get_examples_idx_based_on_type(self, fine_grained_q: dict):
        if fine_grained_q["type"] in self.yes_type_dict:
            no_idx = self.no_type_dict[fine_grained_q["type"]] if self.num_no_examples > 0 else []
            yes_idx = self.yes_type_dict[fine_grained_q["type"]]
            return no_idx, yes_idx
        # There might be multiple types for one question, select the first one
        else:
            for t in self.yes_type_dict.keys():
                if t in fine_grained_q["type"]:
                    return self.no_type_dict[t], self.yes_type_dict[t]
        # the type is "Unknown", use embedding to select examples
        return self.__get_examples_idx_based_on_embedding(fine_grained_q)

    def get_prompts_for_one_pattern_a_time(self, sample: Sample, fine_grained_q: dict):
        """
        Get prompts for one fine-grained question, each prompt corresponds to one guarantee pattern
        """
        prompts = list()
        detailed_query = self.fill_question_slot(
            self.get_sample_detail(sample),
            fine_grained_q["content"],
            fine_grained_q["fn_name"],
            sample.fn_name,
        )
        no_idx, yes_idx = self.__get_examples_idx_based_on_type(fine_grained_q)
        # for idx in yes_idx:
        #     messages = self.prompt.copy()
        #     add_to_message(messages, "user", self.yes_examples[idx]["question"])
        #     add_to_message(messages, "assistant", self.yes_examples[idx]["response"])
        #     for n_idx in no_idx:
        #         add_to_message(messages, "user", self.no_examples[n_idx]["question"])
        #         add_to_message(
        #             messages, "assistant", self.no_examples[n_idx]["response"]
        #         )
        for n_idx in no_idx:
            for y_idx in yes_idx:
                messages = self.prompt.copy()
                add_to_message(messages, "user", self.yes_examples[y_idx]["question"])
                add_to_message(messages, "assistant", self.yes_examples[y_idx]["response"])
                add_to_message(messages, "user", self.no_examples[n_idx]["question"])
                add_to_message(messages, "assistant", self.no_examples[n_idx]["response"])
                add_to_message(messages, "user", detailed_query)
                prompts.append(messages)
        return prompts

    def get_prompt(self, sample: Sample, fine_grained_q: dict):
        """
        Get single prompt for specific fine-grained question
        """
        detailed_query = self.fill_question_slot(
            self.get_sample_detail(sample),
            fine_grained_q["content"],
            fine_grained_q["fn_name"],
            sample.fn_name,
        )
        messages = self.prompt.copy()
        if self.example_enabled:
            if self.criteria == "random":
                no_idx, yes_idx = self.__get_random_idx()
            elif self.criteria == "contract_type":
                no_idx, yes_idx = self.__get_examples_idx_based_on_type(fine_grained_q)
            elif self.criteria == "embedding":
                no_idx, yes_idx = self.__get_examples_idx_based_on_embedding(fine_grained_q)
            for n_idx in no_idx:
                add_to_message(messages, "user", self.no_examples[n_idx]["question"])
                add_to_message(messages, "assistant", self.no_examples[n_idx]["response"])
            for y_idx in yes_idx:
                add_to_message(messages, "user", self.yes_examples[y_idx]["question"])
                add_to_message(messages, "assistant", self.yes_examples[y_idx]["response"])
        add_to_message(messages, "user", detailed_query)
        return messages
