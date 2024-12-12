import asyncio
import json

from tqdm import tqdm

from util.api_stuff import prepare_requests, GenerationModel

with open('scieval-valid.json', 'r') as f:
    data = json.load(f)
# print(data)

with open('llama3_prompt.txt', 'r') as f:
    llama3_prompt_template = f.read()

prepared_prompts = []
for entry in data:
    prepared_prompt = llama3_prompt_template.replace('{problem}', entry['question'])
    prepared_prompt = prepared_prompt.replace('{prompt}', entry['prompt'])
    prepared_prompts.append(prepared_prompt)

batch_size = 250
out = []
for i in tqdm(range(0, len(prepared_prompts), batch_size), desc='Processing SciEval with model LLAMA3',
              total=len(prepared_prompts) // batch_size):
    batch = prepared_prompts[i:i + batch_size]
    output = asyncio.run(
        prepare_requests(
            batch,
            model=GenerationModel.LLAMA3
        )
    )
    for j, entry in enumerate(output):
        # answer from data
        answer = data[i + j]['answer']
        id_ = data[i + j]['id']
        out.append({
            'id': id_,
            'pred': entry['response'],
            'model': 'llama3',
            'prompt': entry['prompt'],
            'category': data[i + j]['category'],
            'ability': data[i + j]['ability'],
            'task_name': data[i + j]['task_name'],
            'answer': data[i + j]['answer'],
            'topic': data[i + j]['topic'],
        })

with open('outputs/llama3_output.json', 'w') as f:
    json.dump(out, f, indent=4)
