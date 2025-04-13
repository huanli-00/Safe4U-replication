## RQ1:
### Meta-Llama-3.1-8B-Instruct
python decompose_safety_and_classify.py --model "Meta-Llama-3.1-8B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U" "basic_check" --model "Meta-Llama-3.1-8B-Instruct" --target "risky" "filtered_unsafe" --device 0
### Qwen2.5-Coder-7B-Instruct
python decompose_safety_and_classify.py --model "Qwen2.5-Coder-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U" "basic_check" --model "Qwen2.5-Coder-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
### Qwen2-7B-Instruct
python decompose_safety_and_classify.py --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U" "basic_check" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
### gpt-4o: use the scripts in `batch_eval_openai.ipynb`

## RQ2:
python batch_eval_vllm.py --prompt "basic+CoT" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U-CoT" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U-hints" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U-references" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U-decompose" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
# python batch_eval_vllm.py --prompt "Safe4U-classified_contract" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U_using_embedding" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U-one_pattern_a_time" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
python batch_eval_vllm.py --prompt "Safe4U-self_check" --model "Qwen2-7B-Instruct" --target "risky" "filtered_unsafe" --device 0
