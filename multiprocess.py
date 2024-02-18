import os
import multiprocessing
from vllm import LLM, SamplingParams

NUM_GPUS = 4


def run_inference_one_gpu(gpu_id, prompt_list, model_name, sampling_params):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model=model_name)
    return llm.generate(prompt_list, sampling_params)


# Splits a list into roughly equally sized pieces
# split_list(["a", "b", "c", "d", "e", "f", "g"], 3) -> [['a', 'b'], ['c', 'd'], ['e', 'f', 'g']]
split_list = lambda l, n: [l[i * len(l) // n : (i + 1) * len(l) // n] for i in range(n)]


def run_inference_multi_gpu(model_name, prompts, sampling_params):
    split_prompts = split_list(prompts, NUM_GPUS)
    inputs = [(i, p, model_name, sampling_params) for i, p in enumerate(split_prompts)]

    with multiprocessing.Pool(processes=NUM_GPUS) as pool:
        results = pool.starmap(run_inference_one_gpu, inputs)

    outputs = []
    for result in results:
        outputs.extend(result)

    return outputs


prompts = [
    f"Write me a story about why {i} is your favourite number.\n\n{i} is my favourite number because "
    for i in range(100)
]
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
model_name = "mistralai/Mistral-7B-v0.2"
outputs = run_inference_multi_gpu(model_name, prompts, sampling_params)
print(outputs)
