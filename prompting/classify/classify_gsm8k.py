import asyncio
import json
import os

import pandas as pd
from datasets import tqdm, load_dataset

from util.api_stuff import prepare_requests, GenerationModel

current_dir = os.path.dirname(os.path.abspath(__file__))

models = ['llama3', 'gpt4', 'claude3']
models = ['bloombert']
models = ['gpt4o']

for model in models:
    print(f'> Processing GSM8K with model {model}')
    if model == 'llama3':
        generation_model = GenerationModel.LLAMA3
        with open(os.path.join(current_dir, '..', '..', 'prompts', 'classify_cognitive_prompt_llama3.txt'), 'r') as f:
            cognitive_prompt_template = f.read()
        with open(os.path.join(current_dir, '..', '..', 'prompts', 'classify_knowledge_prompt_llama3.txt'), 'r') as f:
            knowledge_prompt_template = f.read()
    else:
        with open(os.path.join(current_dir, '..', '..', 'prompts', 'classify_cognitive_prompt_gpt.txt'), 'r') as f:
            cognitive_prompt_template = f.read()
        with open(os.path.join(current_dir, '..', '..', 'prompts', 'classify_knowledge_prompt_gpt.txt'), 'r') as f:
            knowledge_prompt_template = f.read()

        if model == 'gpt4':
            generation_model = GenerationModel.GPT4
        elif model == 'gpt4o':
            generation_model = GenerationModel.GPT4o
        elif model == 'claude3':
            generation_model = GenerationModel.CLAUDE3
        elif model == 'bloombert':
            generation_model = GenerationModel.BLOOMBERT
        else:
            raise NotImplementedError(f'Unknown model {model}')

    dataset = load_dataset("gsm8k", 'main')
    gsm8k_data = dataset['train'][:20]
    out_cognitive = []
    out_knowledge = []
    for question in tqdm(gsm8k_data['question'], desc=f'Processing HumanEval with model {model}',
                         total=len(gsm8k_data['question'])):
        problem = f'{question}'
        cognitive_prompt = cognitive_prompt_template.replace('{problem}', problem)
        knowledge_prompt = knowledge_prompt_template.replace('{problem}', problem)

        output_cognitive = asyncio.run(
            prepare_requests(
                [cognitive_prompt],
                model=generation_model
            )
        )[0]
        if model == 'llama3':
            response_cognitive = output_cognitive['response'] if output_cognitive is not None else ''
        elif model in ['gpt4', 'gpt4o']:
            response_cognitive = output_cognitive['choices'][0]['message'][
                'content'] if output_cognitive is not None else ''
        elif model == 'claude3':
            response_cognitive = output_cognitive['content'][0]['text'] if output_cognitive is not None else ''
        elif model == 'bloombert':
            response_cognitive = output_cognitive['blooms_level'] + '\n'.join(
                f'{k}:{v}' for k, v in output_cognitive['probabilities'].items())
        else:
            raise NotImplementedError(f'Unknown model {model}')
        out_cognitive.append((response_cognitive, cognitive_prompt))

        output_knowledge = asyncio.run(
            prepare_requests(
                [knowledge_prompt],
                model=generation_model
            )
        )[0]
        if model == 'llama3':
            response_knowledge = output_knowledge['response'] if output_knowledge is not None else ''
        elif model in ['gpt4', 'gpt4o']:
            response_knowledge = output_knowledge['choices'][0]['message'][
                'content'] if output_knowledge is not None else ''
        elif model == 'claude3':
            response_knowledge = output_knowledge['content'][0]['text'] if output_knowledge is not None else ''
        elif model == 'bloombert':
            response_knowledge = '> Not supported for Bloombert model <'
        else:
            raise NotImplementedError(f'Unknown model {model}')
        out_knowledge.append((response_knowledge, knowledge_prompt))
    with open(os.path.join(current_dir, 'output_gsm8k', f'{model}_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_gsm8k', f'{model}_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)
