import json
import os

for file in os.listdir('/home/thomas/Downloads/BLOOM/out'):
    if not file.endswith('.json'):
        continue
    if 'agieval' in file:
        task = 'agieval'
    elif 'arc_challenge' in file:
        task = 'arc_challenge'
    elif 'bbh' in file:
        task = 'bbh'
    elif 'boolq' in file:
        task = 'boolq'
    elif 'commonsense_qa' in file:
        task = 'commonsense_qa'
    elif 'drop' in file:
        task = 'drop'
    elif 'gpqa' in file:
        task = 'gpqa'
    elif 'gsm8k' in file:
        task = 'gsm8k'
    elif 'humaneval' in file:
        task = 'humaneval'
    elif 'math' in file:
        task = 'math'
    elif 'quac' in file:
        task = 'quac'
    elif 'squad' in file:
        task = 'squad'
    elif 'triviaqa' in file:
        task = 'triviaqa'
    elif 'winogrande' in file:
        task = 'winogrande'
    else:
        raise Exception('some weird unknown file')

    # rename file, remove '_problems' from the name
    new_name = file.replace('_problems', '')
    # remove '.jsonl' from the name
    new_name = new_name.replace('.jsonl', '')
    # remove '_bbh' from the name
    new_name = new_name.replace('_bbh', '')
    # remove '_mmlu' from the name
    new_name = new_name.replace('_mmlu', '')
    # remove 'output_' from the name
    new_name = new_name.replace('output_', '')
    # remove '.csv' from the name
    new_name = new_name.replace('.csv', '')
    # remove '_agieval' from the name
    new_name = new_name.replace('_agieval', '')
    # remove '.json' from
    new_name = new_name.replace('.json', '')
    new_name = new_name.replace(f'_{task}', '')
    new_name = new_name + '.json'

    with open(f'/home/thomas/Downloads/BLOOM/out/{file}', 'r') as f:
        data = json.load(f)

    out = []
    for gemma_output, gemma_input in data:
        gemma_output = gemma_output[len('<bos>') + len(gemma_input):]
        out.append([gemma_output, gemma_input])

    with open(f'/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/classify/output_{task}/{new_name}', 'w') as f:
        json.dump(out, f, indent=4)