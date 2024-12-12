import asyncio
import json
import os
from typing import List, Optional

from openai import OpenAI

from util.api_stuff import prepare_requests, GenerationModel


async def _make_request_selfhosted(
        prompts: List[str],
        max_tokens: Optional[int] = None,
        model: GenerationModel = GenerationModel.LLAMA
):
    if model == GenerationModel.LLAMA3:
        url = os.environ['LLAMA_OPENAI_ENDPOINT']
        model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
    elif model == GenerationModel.MIXTRAL:
        url = os.environ['MIXTRAL_OPENAI_ENDPOINT']
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    else:
        raise NotImplementedError(f'Unknown model {model}')
    client = OpenAI(
        base_url=url,
        api_key=os.environ['LLAMA_API_KEY']
    )
    batch = client.completions.create(
        model=model_name,
        prompt=prompts,
        max_tokens=max_tokens if max_tokens is not None else 4096,
        temperature=0.0,
    )
    out = []
    for completion in batch.choices:
        out.append({'response': completion.text})
    return out


def check_answers(
        model: str,
        model_answer_dir: str,
):
    model_answer_dir = os.path.join(model_answer_dir, model)
    tasks = sorted(os.listdir(model_answer_dir))

    with open('llama3_prompt_check.txt', 'r') as f:
        llama3_template = f.read()

    try:
        os.mkdir(f'{model}_direct_checked_auto')
    except FileExistsError:
        pass
    try:
        os.mkdir(f'{model}_answers_with_rating')
    except FileExistsError:
        pass

    scores = {}
    checked_tasks = os.listdir(f'{model}_direct_checked_auto')
    checked_tasks = [task for task in checked_tasks if task.endswith('.json')]

    for file in tasks:
        file_name = file.split('_out.json')[0]
        # check if the task has already been checked
        if file_name != 'word_sorting' and f'{file_name}_checked_auto_out.json' in checked_tasks:
            continue

        if file_name not in scores:
            scores[file_name] = {'correct': 0, 'false': 0}
        correct = 0
        false = 0
        with open(os.path.join(model_answer_dir, file), 'r') as f:
            data = json.loads(f.read())

        task_prompts = []
        for llm_solution in data:
            # target = original['target']
            # output = original['output']

            prompt = llama3_template.format(
                task=llm_solution['input'],
                solution=llm_solution['target'],
                student_solution=llm_solution['output']
            )
            task_prompts.append(prompt)

        # outputs = asyncio.run(
        #     prepare_requests(
        #         task_prompts,
        #         model=GenerationModel.LLAMA3,
        #         max_tokens=5
        #     )
        # )
        outputs = asyncio.run(
            _make_request_selfhosted(
                task_prompts,
                model=GenerationModel.LLAMA3,
                max_tokens=5
            )
        )

        auto_ratings = []
        for out in outputs:
            if 'Yes@' in out['response']:
                correct += 1
                auto_ratings.append(True)
            elif 'No@' in out['response']:
                false += 1
                auto_ratings.append(False)
            else:
                if 'yes' in out['response'].lower():
                    correct += 1
                    auto_ratings.append(True)
                elif 'no' in out['response'].lower():
                    false += 1
                    auto_ratings.append(False)
                else:
                    print('No answer found')
                    auto_ratings.append(None)

        scores[file_name]['correct'] += correct
        scores[file_name]['false'] += false

        for task, scores in scores.items():
            accuracy = scores['correct'] / (scores['correct'] + scores['false'])
            print(f'{task}: {accuracy * 100:.2f}')

            out_file_name = f'{task}_checked_auto_out.json'
            with open(os.path.join(f'{model}_direct_checked_auto', out_file_name), 'w') as f:
                json.dump(scores, f, indent=4)

            tmp = []
            for original, auto_score in zip(data, auto_ratings):
                tmp.append({
                    'input': original['input'],
                    'target': original['target'],
                    'output': original['output'],
                    'task_prompt': original['task_prompt'],
                    'auto_score': auto_score
                })
            check_out_file_name = os.path.join(f'{model}_answers_with_rating', f'{file_name}_checked_auto_out.json')
            with open(check_out_file_name, 'w') as f:
                json.dump(tmp, f, indent=4)

        scores = {}


if __name__ == '__main__':
    # check_answers(
    #     model='claude3',
    #     model_answer_dir='/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/bbh_eval/outputs'
    # )

    # # bloomz-3b
    # check_answers(
    #     model='bloomz-3b',
    #     model_answer_dir='/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs/outputs'
    # )

    # bloomz-560m
    # check_answers(
    #     model='bloomz-560m',
    #     model_answer_dir='/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs/outputs'
    # )

    # falcon-7b-instruct
    # check_answers(
    #     model='falcon-7b-instruct',
    #     model_answer_dir='/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs/outputs'
    # )

    # falcon-40b-instruct
    check_answers(
        model='falcon-40b-instruct',
        model_answer_dir='/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs/outputs'
    )

    # flan-t5-xxl
    # check_answers(
    #     model='flan-t5-xxl',
    #     model_answer_dir='/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs/outputs'
    # )

    # # gemma-7b-it
    # check_answers(
    #     model='gemma-7b-it',
    #     model_answer_dir='/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs/outputs'
    # )

    # phi3-mini-4k-instruct
    # check_answers(
    #     model='phi3-mini-4k-instruct',
    #     model_answer_dir='/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs/outputs'
    # )