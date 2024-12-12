import json
import os
import re


def get_scores_old(
        model: str
) -> dict[str, float]:
    current_file_dir = os.path.dirname(__file__)
    model_output_dir = os.path.join(current_file_dir, 'outputs', model)
    tasks = os.listdir(model_output_dir)
    correct_total = 0
    false_total = 0
    task_scores = {}
    for file in tasks:
        correct = 0
        false = 0
        with open(os.path.join(model_output_dir, file), 'r') as f:
            data = json.loads(f.read())

        for row in data:
            target = row['target']
            output = row['output']
            if not output:
                false += 1
                continue
            last_line = [x.strip() for x in output.split('\n') if x.strip()][-1]
            if file == 'word_sorting_out.json':
                # use regex to allow 2 characters in front of each word in the target
                target_words = target.split()
                allowed_regex = r''
                for word in target_words:
                    allowed_regex += '.{0,2}' + word + '.{0,2} '
                allowed_regex = allowed_regex[:-1]
                if output == '':
                    false += 1
                elif not (match := re.search(allowed_regex, output)):
                    false += 1
                else:
                    correct += 1
            else:
                if target in last_line:
                    correct += 1
                else:
                    false += 1
        # print percentage of correct
        # print(f'{file}: {(correct / (correct + false)) * 100}')
        task_name = file.split('_out.json')[0]
        task_scores[task_name] = (correct / (correct + false)) * 100

        correct_total += correct
        false_total += false
    return task_scores


def get_bbh_scores_auto(
        model: str
) -> dict[str, float]:
    current_file_dir = os.path.dirname(__file__)
    model_output_dir = os.path.join(current_file_dir, 'outputs', model)
    # claude3_path = '/prompting/answer_checking/OLD/claude3_checked_auto'
    claude3_path = '/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/answer_checking/OLD/claude3_checked_auto'
    # gpt4_path = '/prompting/answer_checking/OLD/gpt4_checked_auto'
    gpt4_path = '/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/answer_checking/OLD/gpt4_checked_auto'
    # llama3_path = '/prompting/answer_checking/llama3_checked_auto_old'
    llama3_path = '/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/answer_checking/OLD/llama3_checked_auto_old'
    if model == 'claude3':
        model_output_dir = claude3_path
    elif model == 'gpt4':
        model_output_dir = gpt4_path
    elif model == 'llama3':
        model_output_dir = llama3_path
    else:
        model_output_dir = os.path.join('/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/answer_checking', model + '_direct_checked_auto')
        # raise ValueError('Model not found')
    tasks = os.listdir(model_output_dir)
    task_scores = {}
    for file in tasks:
        with open(os.path.join(model_output_dir, file), 'r') as f:
            data = json.loads(f.read())
        task_name = file.split('_checked_auto_out.json')[0]
        if task_name not in task_scores:
            task_scores[task_name] = -1
        accuracy = data['correct'] / (data['correct'] + data['false'])
        task_scores[task_name] = accuracy
    return task_scores


if __name__ == '__main__':
    get_scores_old('claude3')
