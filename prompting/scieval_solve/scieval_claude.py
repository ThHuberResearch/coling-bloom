import asyncio
import json

from tqdm import tqdm

from util.api_stuff import prepare_requests, GenerationModel

with open('scieval-valid.json', 'r') as f:
    data = json.load(f)
# print(data)

with open('gpt_prompt.txt', 'r') as f:
    prompt_template = f.read()

for model in [
    GenerationModel.CLAUDE3
]:
    model_name = model.name.lower()
    prepared_prompts = []
    for entry in data:
        prepared_prompt = prompt_template.replace('{problem}', entry['question'])
        prepared_prompt = prepared_prompt.replace('{prompt}', entry['prompt'])
        prepared_prompts.append(prepared_prompt)

    batch_size = 1
    out = []
    for i in tqdm(range(0, len(prepared_prompts), batch_size), desc=f'Processing SciEval with model {model}',
                  total=len(prepared_prompts) // batch_size):
        batch = prepared_prompts[i:i + batch_size]
        output = asyncio.run(
            prepare_requests(
                batch,
                model=model
            )
        )
        for j, entry in enumerate(output):
            # answer from data
            answer = data[i + j]['answer']
            id_ = data[i + j]['id']
            out.append({
                'id': id_,
                'pred': entry['content'][0]['text'],
                'model': model_name,
                'prompt': prepared_prompts[i + j],
                'category': data[i + j]['category'],
                'ability': data[i + j]['ability'],
                'task_name': data[i + j]['task_name'],
                'answer': data[i + j]['answer'],
                'topic': data[i + j]['topic'],
            })
    with open(f'{model_name}_output.json', 'w') as f:
        json.dump(out, f, indent=4)
