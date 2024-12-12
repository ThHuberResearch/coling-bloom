import asyncio
import json
import os

from tqdm import tqdm

from util.api_stuff import prepare_requests, GenerationModel

tasks = os.listdir('llama3_out')
for task in tasks:
    task_name = task.split('.')[0]
    print(task_name)
    with open(os.path.join('llama3_out', task), 'r') as f:
        data = json.load(f)
    with open('llama3_check_prompt.txt', 'r') as f:
        llama3_prompt_template = f.read()

    prompts = []
    for entry in data:
        output = entry['output']
        target = entry['target']
        question = entry['question']

        prompt = llama3_prompt_template.replace('{task}', question).replace('{solution}', target).replace(
            '{student_solution}', output)
        prompts.append(prompt)

    outputs = asyncio.run(
        prepare_requests(
            prompts,
            model=GenerationModel.LLAMA3,
            max_tokens=4096
        )
    )

    out = []
    for output, entry in zip(outputs, data):
        out.append({
            'solve_prompt': entry['prompt'],
            'target': entry['target'],
            'output': entry['output'],
            'question': entry['question'],
            'answer_check': output["response"]
        })

    with open(os.path.join('llama3_checked', f'{task_name}_out.json'), 'w') as f:
        f.write(json.dumps(out, indent=4))
