# Safe4U

The replication package for paper "Safe4U: Identifying Unsound Safe Encapsulations of Unsafe Calls in Rust using LLMs".

## Content

- 📁 context_retriever: the Rust program that retrieve context of target functions.
- 📁 data: the detailed data
  - 📁 checked: filtered functions to be checked
  - 📁 crate_meta: some meta information of `crates.io`
  - 📁 manual: some data with manual review
  - 📁 samples: functions with fine-grained contracts to be finally checked
- 📁 examples
  - 📄 decompose_and_classify.json: demonstrations for `decomposion & classification` step
  - 📄 document_rewrite.json: rewrite the safety sections that use hyperlink.
  - 📄 examples_for_all_guarantee_patterns.json: examples for `pattern-oriented checks`, including pattern-examples and counter-examples.
- 📁 prompt: prompt and parameter settings
  - 📄 decompose_xxx.json: decomposion & classification
  - 📄 basic_check.json: baseline
  - 📄 Safe4U.json: the complete Safe4U
  - 📄 Safe4U-xxx.json: variant that ablating some components
- 📁 result: the detailed results of evaluation, group by [MODEL/SETTINGS]
- 📁 utils: the utility python code
- 📄 crawl_crates_and_extract_candidates.ipynb
- 📄 sample_check.ipynb
- 📄 decompose_safety_and_classify.py
- 📄 decompose_safety_of_referenced_api.ipynb
- 📄 batch_eval_openai.ipynb
- 📄 batch_eval_vllm.py
- 📄 evaluation_scripts.sh
- 📄 result_viewer.ipynb
- 📄 unsound_functions.md: unsound functions found by Safe4U

## Evaluation

The procedure of Safe4U is separated into several steps:

1. Crawl crates & extract candidate functions with unsafe blocks ([scripts](./crawl_crates_and_extract_candidates.ipynb))
2. Retrieve context with program in `context_retriever`
3. Check and filter the candidate functions according to the context ([scripts](./sample_check.ipynb))
4. Decompose the Safety section into fine-grained classified contracts
5. Check the function by check whether all contracts are guaranteed

All scripts involved in the paper is displayed in `evaluation_scripts.sh`

### Prerequisite

- OS: Linux, e.g., Ubuntu, Debian.
- Environment:
  - python: `python==3.10`
  - python environment: `pip install -r requirements.txt`
  - Rust environment: latest stable tool-chain
- LLM:
  - Using API: OpenAI API key
  - Using local LLM: [vllm](https://docs.vllm.ai/en/latest/getting_started/installation.html) & [open-source LLMs from HuggingFace](https://huggingface.co/models?other=text-generation-inference&sort=trending) (& GPU & CUDA)

### Retrieve Context

The target information are written in `config.toml` with following options:

```toml
[target]
function_locate = "start_line"   <- optional
item_root_path = "/root/dir/for/candidate_info"    # e.g., "data/risky_func"
repo_root_path = "/root/dir/for/repositories"      # e.g., "data/crates_repo"
result_root_path = "/root/dir/for/retrieve_result" # e.g., "data/detailed_risky_func"
```

Run the program with `cargo run target`.
Then, merge and filter the result using [`sample_check.ipynb`](./sample_check.ipynb)

### Decompose Safety

Both OpenAI API and local models are supported in this part.

Run `decompose_safety_and_classify.py` with following options:

```plain
--prompt PROMPT
    Prompt name (e.g., decompose_with_self_check)
--model MODEL
    Model name(OpenAI) or path(Local Models)
--target TARGET [TARGET ...]
    Sample target (risky, filtered_unsafe, 11cve, scan)
--device DEVICE
    CUDA device number
```

### Evaluate with GPT

We use the `Batch` API provided by OpenAI to get 50% discount.
The detailed introduction can be seen in [`batch_eval_openai.ipynb`](./batch_eval_openai.ipynb)

### Evaluate with Local LLM

Similarly, we implement the evaluation with batch to accelerate to calculation speed.

Run `python batch_eval_vllm.py` with following options:

```plain
--prompt [PROMPT ...]
    Prompt name (e.g., Safe4U, basic_check)
--model MODEL
    Model name or path (e.g., Meta-Llama-3.1-8B-Instruct)
--target TARGET [TARGET ...]
    Sample target (e.g., risky, filtered_unsafe, 11cve)
--device DEVICE
    CUDA device number (e.g., 0, 1)
```
