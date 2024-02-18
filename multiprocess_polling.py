import os
import multiprocessing
import time
import requests
from vllm import LLM, SamplingParams

NUM_GPUS = 4


prompts = [
    f"Write me a story about why {i} is your favourite number.\n\n{i} is my favourite number because "
    for i in range(10_000)
]
sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
model_name = "mistralai/Mistral-7b-instruct-v0.2"

def run_inference_one_gpu(gpu_id, prompt_list, model_name, sampling_params):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model=model_name)
    results = []
    while True:       
        full_list = requests.request('GET', 'http://34.31.37.216/llminference', params={'results': results}).json()
        my_list = split_list_i(full_list, NUM_GPUS, gpu_id)
        results = llm.generate(prompt_list, sampling_params)
        time.sleep(.01)


# Splits a list into roughly equally sized pieces
# split_list(["a", "b", "c", "d", "e", "f", "g"], 3) -> [['a', 'b'], ['c', 'd'], ['e', 'f', 'g']]
split_list = lambda l, n: [l[i * len(l) // n : (i + 1) * len(l) // n] for i in range(n)]

split_list_i = lambda l, n, i: [l[i * len(l) // n : (i + 1) * len(l) // n] for i in range(n)][i]


def run_inference_multi_gpu(model_name, prompts, sampling_params):
    split_prompts = split_list(prompts, NUM_GPUS)
    inputs = [(i, p, model_name, sampling_params) for i, p in enumerate(split_prompts)]

    with multiprocessing.Pool(processes=NUM_GPUS) as pool:
        results = pool.starmap(run_inference_one_gpu, inputs)

    outputs = []
    for result in results:
        outputs.extend(result)

    return outputs


outputs = run_inference_multi_gpu(model_name, prompts, sampling_params)
