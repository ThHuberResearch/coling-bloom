import asyncio
import json
import os
import time

from anthropic import Anthropic
from tqdm import tqdm

from util.api_stuff import prepare_requests, GenerationModel

tasks = os.listdir('/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/examples/AGIEval')
tasks = [task for task in tasks if task.endswith('.jsonl')]
for task in tqdm(tasks):
    task_name = task.split('.')[0]
    print(task_name)
    with open(os.path.join('/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/examples/AGIEval', task), 'r') as f:
        data = [json.loads(jline) for jline in f.readlines()]
    with open('claude3_solve_prompt.txt', 'r') as f:
        claude3_solve_prompt = f.read()

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
        prompt = claude3_solve_prompt.replace('{task}', task_full)
        prompts_all.append((prompt, answer, task_full))

    client = Anthropic(
        # This is the default and can be omitted
        api_key=os.environ.get("CLAUDE_API_KEY"),
    )

    outputs = []
    for prompt in tqdm(prompts_all):
        message = client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt[0],
                }
            ],
            model="claude-3-haiku-20240307",
        )
        out = message.content[0].text
        outputs.append(out)

    out = []
    for output, (prompt, answer, question) in zip(outputs, prompts_all):
        response = output
        out.append({
            "output": response,
            "target": answer,
            "prompt": prompt,
            "question": question
        })
    with open(os.path.join('claude3_out', f'{task_name}_out.json'), 'w') as f:
        f.write(json.dumps(out, indent=4))

    # sleep for 1 minute to avoid rate limiting
    time.sleep(60)
