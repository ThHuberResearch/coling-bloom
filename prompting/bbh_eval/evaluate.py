import json
import os
import re

correct_total = 0
false_total = 0
for file in sorted(os.listdir('outputs/gpt4')):
    correct = 0
    false = 0
    with open(os.path.join('outputs/gpt4', file), 'r') as f:
        data = json.loads(f.read())

    for row in data:
        target = row['target']
        output = row['output']
        if file == 'word_sorting_out.json':
            # use regex to allow 2 characters in front of each word in the target
            target_words = target.split()
            allowed_regex = r''
            for word in target_words:
                allowed_regex += '.{0,2}' + word + ' '
            allowed_regex = allowed_regex[:-1]
            if output == '':
                false += 1
            elif not (match := re.search(allowed_regex, output)):
                false += 1
            else:
                correct += 1

        else:
            if target in output:
                correct += 1
            else:
                false += 1
    # print percentage of correct
    print(f'{file}: {(correct / (correct + false)) * 100}')
    correct_total += correct
    false_total += false

print(f'Total: {(correct_total / (correct_total + false_total)) * 100}')