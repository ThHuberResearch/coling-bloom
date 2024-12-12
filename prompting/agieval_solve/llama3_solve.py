import asyncio
import json
import os

from tqdm import tqdm

from util.api_stuff import prepare_requests, GenerationModel

tasks = os.listdir('/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/examples/AGIEval')
tasks = [task for task in tasks if task.endswith('.jsonl')]
for task in tqdm(tasks):
    task_name = task.split('.')[0]
    print(task_name)
    with open(os.path.join('/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/examples/AGIEval', task), 'r') as f:
        data = [json.loads(jline) for jline in f.readlines()]
    with open('llama3_solve_prompt.txt', 'r') as f:
        llama3_prompt_template = f.read()

    prompts_all = []
    for task in data:
        passage = task['passage']
        passage = '' if passage is None else passage
        try:
            options = '\n'.join(task['options'])
            answer = task['label']
        except TypeError:
            options = ''
            answer = task['answer']
        task_full = f'{passage}\n{task["question"]}\n{options}'
        prompt = llama3_prompt_template.replace('{task}', task_full)
        prompts_all.append((prompt, answer, task_full))
    outputs = asyncio.run(
        prepare_requests(
            [prompt for prompt, answer, question in prompts_all],
            model=GenerationModel.LLAMA3,
            max_tokens=4096
        )
    )

    out = []
    for output, (_prompt, answer, question) in zip(outputs, prompts_all):
        response = output["response"] if output is not None else ""
        prompt = output["prompt"] if output is not None else ""
        out.append({
            "output": response,
            "target": answer,
            "prompt": prompt,
            "question": question
        })
    with open(os.path.join('llama3_out', f'{task_name}_out.json'), 'w') as f:
        f.write(json.dumps(out, indent=4))
