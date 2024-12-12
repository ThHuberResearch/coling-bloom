import json
import os

result_path = '/prompting/answer_checking/OLD/gpt4_checked'

scores = {}
for file in sorted(os.listdir(result_path)):
    file_name = file.split('_out.json')[0]
    if file.endswith('.json'):
        with open(os.path.join(result_path, file), 'r') as f:
            data = json.loads(f.read())

        correct = 0
        wrong = 0
        for entry in data:
            if 'auto_correct' in entry:
                if entry['auto_correct']:
                    correct += 1
                else:
                    wrong += 1
            elif 'user_correct' in entry:
                if entry['user_correct']:
                    correct += 1
                else:
                    wrong += 1
            else:
                print('no auto_correct or human_correct in file')
        if file_name not in scores:
            scores[file_name] = {}
        scores[file_name]['correct'] = correct
        scores[file_name]['wrong'] = wrong

for task, scores in scores.items():
    accuracy = scores['correct'] / (scores['correct'] + scores['wrong'])
    print(f'{task}: {accuracy*100:.2f}')
