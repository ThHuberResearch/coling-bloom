import json
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

tasks = [
    'agieval',
    'arc_challenge',
    'boolq',
    'commonsense_qa',
    'drop',
    'gpqa',
    'gsm8k',
    'humaneval',
    'math',
    # 'mmlu',
    'quac',
    'squad',
    'triviaqa',
    'winogrande'
]

cognitive = [
    'Create',
    'Evaluate',
    'Analyze',
    'Apply',
    'Understand',
    'Remember'
]
knowledge = [
    'Metacognitive',
    'Procedural',
    'Conceptual',
    'Factual'
]


def parse_output(
        output: str,
        type_: str,
):
    if type_ == 'knowledge':
        relevant_classes = knowledge
    elif type_ == 'cognitive':
        relevant_classes = cognitive
    else:
        raise ValueError(f'Invalid type: {type_}')
    for cls in relevant_classes:
        if cls in output:
            return cls
    else:
        for cls in relevant_classes:
            if cls.lower() in output.lower():
                return cls
        else:
            # print(f'No class found in output: {output}')
            return None


def agieval_results() -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    agieval_files = os.listdir(os.path.join('classify', 'output_agieval'))
    output_cognitive = []
    output_knowledge = []
    for agieval_file in agieval_files:
        model, task, type_ = agieval_file.split('_')
        if model == 'bloombert':
            continue
        if task == 'sat-en-without-passage':
            continue  # the useless one, ignore it
        type_ = type_.split('.')[0]
        with open(os.path.join('classify', 'output_agieval', agieval_file)) as f:
            data = json.loads(f.read())
        for output, _prompt in data:
            cls = parse_output(output, type_)
            # if cls is not None:
            if type_ == 'cognitive':
                output_cognitive.append((model, 'AGIEval', task, cls))
            elif type_ == 'knowledge':
                output_knowledge.append((model, 'AGIEval', task, cls))
    return output_cognitive, output_knowledge


def get_task_results(
        task: str,
        task_name: str
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    task_files = os.listdir(os.path.join('classify', f'output_{task}'))
    output_cognitive = []
    output_knowledge = []
    for task_file in task_files:
        model, type_ = task_file.split('_')
        if model == 'bloombert':
            continue
        type_ = type_.split('.')[0]
        with open(os.path.join('classify', f'output_{task}', task_file)) as f:
            data = json.loads(f.read())
        for output, _prompt in data:
            cls = parse_output(output, type_)
            # if cls is not None:
            if type_ == 'cognitive':
                output_cognitive.append((model, task_name, 'N/A', cls))
            elif type_ == 'knowledge':
                output_knowledge.append((model, task_name, 'N/A', cls))
    return output_cognitive, output_knowledge


def math_results() -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    math_files = os.listdir(os.path.join('classify', 'output_math'))
    output_cognitive = []
    output_knowledge = []
    for math_file in math_files:
        model, *task, type_ = math_file.split('_')
        if model == 'bloombert':
            continue
        task = '_'.join(task)
        type_ = type_.split('.')[0]
        with open(os.path.join('classify', 'output_math', math_file)) as f:
            data = json.loads(f.read())
        for output, _prompt in data:
            cls = parse_output(output, type_)
            # if cls is not None:
            # decided to ignore subtasks, because they are extremely similar, and instead do majority voting across the subtasks for result
            if type_ == 'cognitive':
                output_cognitive.append((model, 'MATH', 'N/A', cls))
            elif type_ == 'knowledge':
                output_knowledge.append((model, 'MATH', 'N/A', cls))
    return output_cognitive, output_knowledge


def do_majority_voting(
        cognitive: list[tuple[str, str, str, str]],
        knowledge: list[tuple[str, str, str, str]],
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    cognitive_dict = {}
    knowledge_dict = {}
    for model, task, subtask, cls in cognitive:
        if (model, task, subtask) not in cognitive_dict:
            cognitive_dict[(model, task, subtask)] = {}
        if cls not in cognitive_dict[(model, task, subtask)]:
            cognitive_dict[(model, task, subtask)][cls] = 0
        cognitive_dict[(model, task, subtask)][cls] += 1
    for model, task, subtask, cls in knowledge:
        if (model, task, subtask) not in knowledge_dict:
            knowledge_dict[(model, task, subtask)] = {}
        if cls not in knowledge_dict[(model, task, subtask)]:
            knowledge_dict[(model, task, subtask)][cls] = 0
        knowledge_dict[(model, task, subtask)][cls] += 1
    cognitive = []
    knowledge = []
    for (model, task, subtask), cls_dict in cognitive_dict.items():
        cls = max(cls_dict, key=cls_dict.get)
        cognitive.append((model, task, subtask, cls))
    for (model, task, subtask), cls_dict in knowledge_dict.items():
        cls = max(cls_dict, key=cls_dict.get)
        knowledge.append((model, task, subtask, cls))
    cognitive = sorted(cognitive, key=lambda x: (x[0], x[1], x[2]))
    knowledge = sorted(knowledge, key=lambda x: (x[0], x[1], x[2]))
    return cognitive, knowledge


def human_results():
    human_df = pd.read_excel('EMNLP Bloom Taxonomy Classification.xlsx')

    # replace NaN with 'N/A'
    human_df = human_df.fillna('N/A')
    return human_df


def human_results_v2():
    human_df = pd.read_excel('EMNLP Bloom Taxonomy Classification_v2.xlsx')

    # replace NaN with 'N/A'
    human_df = human_df.fillna('N/A')
    return human_df


def human_results_v3():
    human_df = pd.read_excel('EMNLP Bloom Taxonomy Classification_v3.xlsx')

    # replace NaN with 'N/A'
    human_df = human_df.fillna('N/A')
    return human_df


def human_results_v4():
    human_df = pd.read_excel('EMNLP Bloom Taxonomy Classification_v4.xlsx')

    # replace NaN with 'N/A'
    human_df = human_df.fillna('N/A')
    return human_df


def bbh_results():
    bbh_files = os.listdir(os.path.join('classify', 'output_bbh'))
    output_cognitive = []
    output_knowledge = []
    for bbh_file in bbh_files:
        model, *task, type_ = bbh_file.split('_')
        type_ = type_.split('.')[0]
        if model == 'bloombert':
            continue

        # comment this OUT to include moding model!!
        if model == 'voting':
            continue
        task = '_'.join(task)
        with open(os.path.join('classify', 'output_bbh', bbh_file)) as f:
            data = json.loads(f.read())
        for output, _prompt in data:
            cls = parse_output(output, 'knowledge')
            # if cls is not None:
            if type_ == 'knowledge':
                output_knowledge.append((model, 'Big Bench Hard', task, cls))
            cls = parse_output(output, 'cognitive')
            # if cls is not None:
            if type_ == 'cognitive':
                output_cognitive.append((model, 'Big Bench Hard', task, cls))
    return output_cognitive, output_knowledge


if __name__ == '__main__':
    df = pd.DataFrame(columns=['Model', 'Benchmark / Dataset', 'Subtask', 'Cognitive', 'Knowledge'])
    results = [
        agieval_results(),
        get_task_results('arc_challenge', 'ARC-Challenge'),
        get_task_results('boolq', 'BoolQ'),
        get_task_results('commonsense_qa', 'CommonsenseQA'),
        get_task_results('drop', 'DROP'),
        get_task_results('gpqa', 'GPQA'),
        get_task_results('gsm8k', 'GSM8K'),
        get_task_results('humaneval', 'HumanEval'),
        get_task_results('quac', 'QuAC'),
        get_task_results('squad', 'SQuAD'),
        get_task_results('triviaqa', 'TriviaQA'),
        get_task_results('winogrande', 'Winogrande'),
        math_results(),
        bbh_results()
    ]
    for cognitive, knowledge in results:
        cognitive, knowledge = do_majority_voting(cognitive, knowledge)
        for (model_cognitive, task_cognitive, subtask_cognitive, cls_cognitive), (
                model_knowledge, task_knowledge, subtask_knowledge, cls_knowledge) in zip(cognitive, knowledge):
            if model_cognitive != model_knowledge:
                raise ValueError(f'Mismatched models: {model_cognitive} and {model_knowledge}')
            if task_cognitive != task_knowledge:
                raise ValueError(f'Mismatched tasks: {task_cognitive} and {task_knowledge}')
            if subtask_cognitive != subtask_knowledge:
                raise ValueError(f'Mismatched subtasks: {subtask_cognitive} and {subtask_knowledge}')
            df = pd.concat([
                df,
                pd.DataFrame(
                    {
                        'Model': [model_cognitive],
                        'Benchmark / Dataset': [task_cognitive],
                        'Subtask': [subtask_cognitive],
                        'Cognitive': [cls_cognitive],
                        'Knowledge': [cls_knowledge]
                    }
                )
            ])
    # v1: original
    # v2: once we cleared up reading comprehension
    # v3: checking everything again after checking results and classifying the Cognitive again (knowledge is same)
    # v4: updates from christina incorporated
    human_df = human_results_v4()
    for i, row in human_df.iterrows():
        df = pd.concat([
            df,
            pd.DataFrame(
                {
                    'Model': ['Human'],
                    'Benchmark / Dataset': [row['Benchmark / Dataset']],
                    'Subtask': [row['Subtask']],
                    'Cognitive': [row['Cognitive']],
                    'Knowledge': [row['Knowledge']]
                }
            )
        ])

    # exclude those where model == 'gemma'
    df = df[df['Model'] != 'gemma']
    # also exclude bloomberta
    df = df[df['Model'] != 'bloomberta']

    pivot_cognitive = df.pivot_table(index=['Benchmark / Dataset', 'Subtask'], columns='Model', values='Cognitive',
                                     aggfunc='first')
    pivot_knowledge = df.pivot_table(index=['Benchmark / Dataset', 'Subtask'], columns='Model', values='Knowledge',
                                     aggfunc='first')

    # drop rows where Dataset / Benchmark is one of BoolQ, QuAC, SQuAD, TriviaQA, CommonsenseQA
    pivot_cognitive = pivot_cognitive[~pivot_cognitive.index.get_level_values('Benchmark / Dataset').isin(
        ['BoolQ', 'QuAC', 'SQuAD', 'TriviaQA', 'CommonsenseQA'])]
    pivot_knowledge = pivot_knowledge[~pivot_knowledge.index.get_level_values('Benchmark / Dataset').isin(
        ['BoolQ', 'QuAC', 'SQuAD', 'TriviaQA', 'CommonsenseQA'])]

    # dropna to remove rows with missing values
    pivot_cognitive = pivot_cognitive.dropna()
    pivot_knowledge = pivot_knowledge.dropna()

    models = df['Model'].unique()

    # Prepare to store kappa scores
    kappa_scores_cognitive = {}
    kappa_scores_knowledge = {}

    # Calculate kappa for each pair of models
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_i = models[i]
            model_j = models[j]
            kappa_cog = cohen_kappa_score(pivot_cognitive[model_i], pivot_cognitive[model_j])
            try:
                kappa_know = cohen_kappa_score(pivot_knowledge[model_i], pivot_knowledge[model_j])
            except KeyError:
                # set to NaN
                kappa_know = float('nan')
            kappa_scores_cognitive[(model_i, model_j)] = kappa_cog
            kappa_scores_knowledge[(model_i, model_j)] = kappa_know

    kappa_scores_cognitive = dict(sorted(kappa_scores_cognitive.items(), key=lambda item: item[1], reverse=True))
    kappa_scores_knowledge = dict(sorted(kappa_scores_knowledge.items(), key=lambda item: item[1], reverse=True))

    # print(f'======== Unweighted Kappa Scores ========')
    # for pair, score in kappa_scores_cognitive.items():
    #     print(f'Cognitive Agreement between {pair[0]} and {pair[1]}: {score}')
    # for pair, score in kappa_scores_knowledge.items():
    #     print(f'Knowledge Agreement between {pair[0]} and {pair[1]}: {score}')
    # print()

    models = df['Model'].unique()
    kappa_scores_cognitive = {}
    kappa_scores_knowledge = {}

    # Define quadratic weights
    weights = 'quadratic'

    knowledge_mapping = {
        'Metacognitive': 0,
        'Procedural': 1,
        'Conceptual': 2,
        'Factual': 3
    }
    cognitive_mapping = {
        'Create': 0,
        'Evaluate': 1,
        'Analyze': 2,
        'Apply': 3,
        'Understand': 4,
        'Remember': 5
    }

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_i = models[i]
            model_j = models[j]
            model_i_mapped_cognitive = pivot_cognitive[model_i].map(cognitive_mapping)
            model_j_mapped_cognitive = pivot_cognitive[model_j].map(cognitive_mapping)

            try:
                model_i_mapped_knowledge = pivot_knowledge[model_i].map(knowledge_mapping)
                model_j_mapped_knowledge = pivot_knowledge[model_j].map(knowledge_mapping)
                kappa_know = cohen_kappa_score(model_i_mapped_knowledge, model_j_mapped_knowledge, weights=weights)
            except KeyError:
                kappa_know = float('nan')

            kappa_cog = cohen_kappa_score(model_i_mapped_cognitive, model_j_mapped_cognitive, weights=weights)
            kappa_scores_cognitive[(model_i, model_j)] = kappa_cog
            kappa_scores_knowledge[(model_i, model_j)] = kappa_know

    kappa_scores_cognitive = dict(sorted(kappa_scores_cognitive.items(), key=lambda item: item[1], reverse=True))
    kappa_scores_knowledge = dict(sorted(kappa_scores_knowledge.items(), key=lambda item: item[1], reverse=True))

    models_all = df['Model'].unique()

    print(f'======== Weighted Kappa Scores ({weights} weights) ========')
    for pair, score in kappa_scores_cognitive.items():
        print(f'Cognitive Agreement between {pair[0]} and {pair[1]}: {score}')
    for pair, score in kappa_scores_knowledge.items():
        print(f'Knowledge Agreement between {pair[0]} and {pair[1]}: {score}')

    models_all = df['Model'].unique()
    # ensure 'Human' is first and rest is alphabetical
    model_order = ['Human'] + sorted([model for model in models_all if model != 'Human'])

    table = '''\\begin{table*}[t]'''
    table += '''
\\centering
\\begin{tabular}''' + '{l' + 'r' * len(model_order) + '}' + '''
\\toprule
'''
    table += '''\\textbf{} & ''' + ' & '.join([f'\\textbf{{{model}}}' for model in model_order]) + ''' \\\\ \\toprule'''
    model2agreement = {
        model: {model2: 0 for model2 in model_order if model2 != model} for model in model_order
    }

    for pair, score in kappa_scores_cognitive.items():
        model1, model2 = pair
        model2agreement[model1][model2] = score
        model2agreement[model2][model1] = score

    # calculate average agreement for each model
    for model in model_order:
        total = 0
        count = 0
        for model2 in model_order:
            if model == model2:
                continue
            total += model2agreement[model][model2]
            count += 1
        print(f'Average agreement (cognitive) for {model}: {total / count}')

    for model in model_order:
        table += f'\n{model} & '
        for model2 in model_order:
            if model == model2:
                table += '- & '
            else:
                table += f'{model2agreement[model][model2]:.2f} & '
        table = table[:-2]
        table += ' \\\\'
    table += '''\\bottomrule
\\end{tabular}
\\caption{Inter-model agreement for Cognitive Dimension}
\\label{tab:agreement}
\\end{table*}'''
    print(table)

    print('\n\n')

    # make second table, for knowledge
    table = '''\\begin{table*}[t]'''
    table += '''
\\centering
\\begin{tabular}''' + '{l' + 'r' * len(model_order) + '}' + '''
\\toprule
'''
    table += '''\\textbf{} & ''' + ' & '.join([f'\\textbf{{{model}}}' for model in model_order]) + ''' \\\\ \\toprule'''
    model2agreement = {
        model: {model2: 0 for model2 in model_order if model2 != model} for model in model_order
    }

    for pair, score in kappa_scores_knowledge.items():
        model1, model2 = pair
        model2agreement[model1][model2] = score
        model2agreement[model2][model1] = score

    # calculate average agreement for each model
    for model in model_order:
        total = 0
        count = 0
        for model2 in model_order:
            if model == model2:
                continue
            total += model2agreement[model][model2]
            count += 1
        print(f'Average agreement (knowledge) for {model}: {total / count}')

    for model in model_order:
        table += f'\n{model} & '
        for model2 in model_order:
            if model == model2:
                table += '- & '
            else:
                table += f'{model2agreement[model][model2]:.2f} & '
        table = table[:-2]
        table += ' \\\\'
    table += '''\\bottomrule
\\end{tabular}
\\caption{Inter-model agreement for Knowledge Types}
\\label{tab:agreement_knowledge}
\\end{table*}'''
    print(table)

    # currently voting is commented OUT in bbh_results!!
