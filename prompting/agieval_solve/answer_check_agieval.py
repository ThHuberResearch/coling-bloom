import asyncio
import json
import os

from prompting.answer_checking.answer_check_with_llama3 import _make_request_selfhosted
from util.api_stuff import GenerationModel


def check_answers(
        model: str,
        model_answer_dir: str
):
    model_answer_dir = os.path.join(model_answer_dir, model)
    tasks = sorted(os.listdir(model_answer_dir))

    with open('llama3_check_prompt.txt', 'r') as f:
        llama3_template = f.read()

    try:
        os.mkdir(f'{model}_out')
    except FileExistsError:
        pass
    try:
        os.mkdir(f'{model}_checked')
    except FileExistsError:
        pass

    scores = {}
    checked_tasks = os.listdir(f'{model}_out')
    checked_tasks = [task for task in checked_tasks if task.endswith('.json')]

    for file in tasks:
        file_name = file.split('_out.json')[0]
        # check if the task has already been checked
        if f'{file_name}_out.json' in checked_tasks:
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

        # check the answers
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

            out_file_name = f'{task}_out.json'
            with open(os.path.join(f'{model}_checked', out_file_name), 'w') as f:
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
            check_out_file_name = os.path.join(f'{model}_out', f'{file_name}_out.json')
            with open(check_out_file_name, 'w') as f:
                json.dump(tmp, f, indent=4)

        scores = {}


if __name__ == '__main__':
    # check_answers(
    #     'bloomz-3b',
    #     '/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs_agieval/outputs_agieval'
    # )
    _models = [
        'bloomz-560m',
        'falcon-7b-instruct',
        'falcon-40b-instruct',
        'flan-t5-small',
        'flan-t5-xxl',
        'gemma-7b-it',
        'phi3-mini-4k-instruct'
    ]
    for model in _models:
        check_answers(
            model=model,
            model_answer_dir='/home/thomas/PycharmProjects/protefra/projects/bloom_eval_paper/outputs_agieval/outputs_agieval'
        )