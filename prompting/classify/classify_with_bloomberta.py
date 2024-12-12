import json
import os

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

labels = ['Analyze', 'Apply', 'Create', 'Evaluate', 'Remember', 'Understand']

model = AutoModelForSequenceClassification.from_pretrained(
    '/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/bloomberta')
tokenizer = AutoTokenizer.from_pretrained('/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/bloomberta')

model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def classify_with_bloomberta(texts):
    tokenized_datasets = tokenizer(texts, truncation=True, padding=True)

    input_ids = torch.tensor(tokenized_datasets['input_ids'])
    attention_mask = torch.tensor(tokenized_datasets['attention_mask'])

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20)

    predictions = []
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())

    predicted_labels = [labels[pred] for pred in predictions]

    return predicted_labels


# agieval
current_dir = os.path.dirname(os.path.abspath(__file__))


def classify_agieval():
    tasks = os.listdir(os.path.join(current_dir, '..', '..', 'examples', 'AGIEval'))
    for task in tqdm(tasks, desc=f'Processing AGIEval with model {model}', total=len(tasks)):
        task_name = task.split('.')[0]
        with open(os.path.join(current_dir, '..', '..', 'examples', 'AGIEval', task), 'r') as f:
            lines = [json.loads(line) for line in f.readlines()]
        samples = lines[:20]
        texts = []
        for sample in samples:
            passage = ''
            if sample['passage'] is not None:
                passage = sample['passage']
            options = []
            if sample['options'] is not None:
                options = sample['options']
            problem = f'{passage}\n{sample["question"]}\n{" ".join(options)}'
            texts.append(problem)
        classes = classify_with_bloomberta(texts)
        out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
        out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]

        with open(os.path.join(current_dir, 'output_agieval', f'bloomberta_{task_name}_cognitive.json'), 'w') as f:
            json.dump(out_cognitive, f)
        with open(os.path.join(current_dir, 'output_agieval', f'bloomberta_{task_name}_knowledge.json'), 'w') as f:
            json.dump(out_knowledge, f)


def classify_drop():
    drop_path = os.path.join(current_dir, '..', '..', 'examples', 'DROP', 'drop_dataset', 'drop_dataset_train.json')
    with open(drop_path, 'r') as f:
        drop = json.load(f)
    drop = list(drop.items())[:20]
    texts = []
    for _id, data in tqdm(drop, desc=f'Processing Winogrande with model {model}', total=len(drop)):
        passage = data['passage']
        qa_pairs = data['qa_pairs']
        relevant_qa = qa_pairs[0]
        question = relevant_qa['question']
        problem = f'{question}\nExtract the answer from the following context:\n{passage}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_drop', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_drop', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_gpqa():
    dataset = load_dataset("Idavidrein/gpqa", 'gpqa_diamond')
    gpqa_data = dataset['train'][:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for question in tqdm(gpqa_data['Question'], total=len(gpqa_data['Question'])):
        problem = f'{question}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_gpqa', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_gpqa', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_gsm8k():
    dataset = load_dataset("gsm8k", 'main')
    gsm8k_data = dataset['train'][:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for question in tqdm(gsm8k_data['question'], desc=f'Processing HumanEval with model {model}',
                         total=len(gsm8k_data['question'])):
        problem = f'{question}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_gsm8k', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_gsm8k', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_math():
    tasks = sorted(os.listdir(os.path.join(current_dir, '..', '..', 'examples', 'MATH', 'MATH', 'train')))
    for task in tqdm(tasks, desc=f'Processing MATH with model {model}', total=len(tasks)):
        task_files = sorted(
            os.listdir(os.path.join(current_dir, '..', '..', 'examples', 'MATH', 'MATH', 'train', task)))
        task_files = task_files[:20]

        out_knowledge = []
        out_cognitive = []
        texts = []
        for task_file in task_files:
            with open(os.path.join(current_dir, '..', '..', 'examples', 'MATH', 'MATH', 'train', task, task_file),
                      'r') as f:
                task_json = json.load(f)
            problem = f'{task_json["problem"]}'
            texts.append(problem)
        classes = classify_with_bloomberta(texts)
        out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
        out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
        with open(os.path.join(current_dir, 'output_math', f'bloomberta_{task}_cognitive.json'), 'w') as f:
            json.dump(out_cognitive, f)
        with open(os.path.join(current_dir, 'output_math', f'bloomberta_{task}_knowledge.json'), 'w') as f:
            json.dump(out_knowledge, f)


def classify_humaneval():
    dataset = load_dataset("openai_humaneval")
    humaneval_data = dataset['test'][:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for prompt in tqdm(humaneval_data['prompt'], desc=f'Processing HumanEval with model {model}',
                       total=len(humaneval_data['prompt'])):
        problem = f'Write code for the following function:\n{prompt}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_humaneval', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_humaneval', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_mmlu():
    tasks = sorted(os.listdir(os.path.join(current_dir, '..', '..', 'examples', 'mmlu')))
    for task in tqdm(tasks, desc=f'Processing MATH with model {model}', total=len(tasks)):
        task_name = task.split('.')[0]
        out_knowledge = []
        out_cognitive = []
        texts = []
        task_df = pd.read_csv(os.path.join(current_dir, '..', '..', 'examples', 'mmlu', task))
        task_df = task_df[:20]
        for i, row in task_df.iterrows():
            task_json = row.to_dict()
            problem = f'{task_json["input"]}'
            texts.append(problem)
        classes = classify_with_bloomberta(texts)
        out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
        out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
        with open(os.path.join(current_dir, 'output_mmlu', f'bloomberta_{task_name}_cognitive.json'), 'w') as f:
            json.dump(out_cognitive, f)
        with open(os.path.join(current_dir, 'output_mmlu', f'bloomberta_{task_name}_knowledge.json'), 'w') as f:
            json.dump(out_knowledge, f)


def classify_quac():
    trivia_qa_path = os.path.join(current_dir, '..', '..', 'examples', 'SQuAD', 'train-v2.0.json')
    with open(trivia_qa_path, 'r') as f:
        quac = json.load(f)
    quac_data = quac['data'][:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for data in tqdm(quac_data, desc=f'Processing QuAC with model {model}', total=len(quac_data)):
        title = data['title']
        paragraphs = data['paragraphs']
        relevant_paragraph = paragraphs[0]
        context = relevant_paragraph['context']
        qas = relevant_paragraph['qas']
        relevant_q = qas[0]
        question = relevant_q['question']
        problem = f'{question}\nExtract the answer from the following context:\n{context}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_quac', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_quac', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_squad():
    trivia_qa_path = os.path.join(current_dir, '..', '..', 'examples', 'SQuAD', 'train-v2.0.json')
    with open(trivia_qa_path, 'r') as f:
        squad = json.load(f)
    squad_data = squad['data'][:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for data in tqdm(squad_data, desc=f'Processing Winogrande with model {model}', total=len(squad_data)):
        title = data['title']
        paragraphs = data['paragraphs']
        relevant_paragraph = paragraphs[0]
        context = relevant_paragraph['context']
        qas = relevant_paragraph['qas']
        relevant_q = qas[0]
        question = relevant_q['question']
        problem = f'{question}\nExtract the answer from the following context:\n{context}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_squad', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_squad', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_triviaqa():
    trivia_qa_path = os.path.join(current_dir, '..', '..', 'examples', 'TriviaQA', 'wikipedia-train.json')
    with open(trivia_qa_path, 'r') as f:
        trivia_qa = json.load(f)
    trivia_qa_data = trivia_qa['Data'][:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for data in tqdm(trivia_qa_data, desc=f'Processing Winogrande with model {model}', total=len(trivia_qa_data)):
        question = data['Question']
        num_pages = len(data['EntityPages'])
        problem = f'{question}\nThe answer can be found in one of {num_pages} pages. They are omitted here.'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_triviaqa', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_triviaqa', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_winogrande():
    winogrande_df = pd.read_csv(os.path.join(current_dir, '..', '..', 'examples', 'winogrande', 'winogrande_xl.csv'))
    winogrande_df = winogrande_df.iloc[:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for i, row in tqdm(winogrande_df.iterrows(), desc=f'Processing Winogrande with model {model}',
                       total=len(winogrande_df)):
        problem = f'{row["sentence"]}\n{row["option1"]} - {row["option2"]}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_winogrande', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_winogrande', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_bbh():
    tasks = sorted(os.listdir(os.path.join(current_dir, '..', '..', 'examples', 'bbh')))

    for task in tqdm(tasks, desc=f'Processing MATH with model {model}', total=len(tasks)):
        task_name = task.split('.')[0]
        with open(os.path.join(current_dir, '..', '..', 'examples', 'bbh', task), 'r') as f:
            lines = [json.loads(line) for line in f.readlines()]
        data = lines[0]['examples'][:20]

        out_knowledge = []
        out_cognitive = []
        texts = []
        for problem in data:
            problem = f'{problem["input"]}'
            texts.append(problem)
        classes = classify_with_bloomberta(texts)
        out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
        out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
        with open(os.path.join(current_dir, 'output_bbh', f'bloomberta_{task_name}_cognitive.json'), 'w') as f:
            json.dump(out_cognitive, f)
        with open(os.path.join(current_dir, 'output_bbh', f'bloomberta_{task_name}_knowledge.json'), 'w') as f:
            json.dump(out_knowledge, f)


def classify_commonsense_qa():
    commonsense_df = pd.read_csv(
        os.path.join(current_dir, '..', '..', 'examples', 'commonsense_qa', 'commonsense_qa.csv'))
    commonsense_df = commonsense_df.iloc[:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for i, row in tqdm(commonsense_df.iterrows(), desc=f'Processing CommonSenseQA with model {model}',
                       total=len(commonsense_df)):
        problem = f'{row["question"]}\n{row["choices"]}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_commonsense_qa', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_commonsense_qa', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_arc_challenge():
    arc_challenge_path = os.path.join(current_dir, '..', '..', 'examples', 'arc_challenge', 'arc_challenge.csv')
    arc_challenge = pd.read_csv(arc_challenge_path)
    arc_df = pd.read_csv(os.path.join(current_dir, '..', '..', 'examples', 'arc_challenge', 'arc_challenge.csv'))
    arc_df = arc_df.iloc[:20]
    texts = []
    for i, row in tqdm(arc_df.iterrows(), desc=f'Processing ARC-Challenge with model {model}', total=len(arc_df)):
        problem = f'{row["question"]}\n{row["choices"]}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_arc_challenge', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_arc_challenge', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_boolq():
    dataset = load_dataset("google/boolq")
    boolq_data = dataset['train'][:20]
    out_cognitive = []
    out_knowledge = []
    texts = []
    for question, answer, passage in tqdm(zip(boolq_data['question'], boolq_data['answer'], boolq_data['passage']),
                                          desc=f'Processing BoolQ with model {model}', total=len(boolq_data)):
        problem = f'{question}\nExtract the answer from the following context:\n{passage}'
        texts.append(problem)
    classes = classify_with_bloomberta(texts)
    out_cognitive = [(cls, sample) for cls, sample in zip(classes, texts)]
    out_knowledge = [('', sample) for cls, sample in zip(classes, texts)]
    with open(os.path.join(current_dir, 'output_boolq', f'bloomberta_cognitive.json'), 'w') as f:
        json.dump(out_cognitive, f)
    with open(os.path.join(current_dir, 'output_boolq', f'bloomberta_knowledge.json'), 'w') as f:
        json.dump(out_knowledge, f)


def classify_from_descriptions():
    descriptions_file = '/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/bbh_descriptions.txt'
    with open(descriptions_file, 'r') as f:
        descriptions = f.readlines()
    tasks = [desc.split(':') for desc in descriptions]
    tasks = [(task[0], ':'.join(task[1:]).strip()) for task in tasks]
    for task, desc in tasks:
        cls = classify_with_bloomberta([desc])[0]
        print(f'{task}: {cls}')


# classify_arc_challenge()
# classify_from_descriptions()
classify_boolq()
# classify_agieval()
# classify_drop()
# classify_gpqa()
# classify_gsm8k()
# classify_math()
# classify_humaneval()
# classify_mmlu()
# classify_quac()
# classify_squad()
# classify_triviaqa()
# classify_winogrande()
# classify_bbh()
# classify_commonsense_qa()
