import json
import os

files = sorted(os.listdir('claude3_checked'))
scores = {}
for file in files:
    task_name = file.split('_out_out.')[0]

    with open(os.path.join('claude3_checked', file), 'r') as f:
        data = json.load(f)

    correct = 0
    wrong = 0
    for item in data:
        if 'yes@' in item['answer_check'].lower():
            correct += 1
        elif 'no@' in item['answer_check'].lower():
            wrong += 1
        else:
            if 'no' in item['answer_check'].lower():
                wrong += 1
                continue
            # god, why
            print(item['answer_check'])
    accuracy = correct / (correct + wrong)
    if task_name not in scores:
        scores[task_name] = -1
    scores[task_name] = accuracy

print(scores)