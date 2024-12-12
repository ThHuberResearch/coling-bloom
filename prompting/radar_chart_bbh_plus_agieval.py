import json
import os

import numpy as np
import pandas as pd

from prompting.bbh_eval.bbh_scores import get_bbh_scores_auto
import pandas as pd
import plotly.io as pio
import plotly.express as px

from prompting.score_mapping_and_chart import map_bbh_scores, get_excel_task_mapping, map_other_tasks

# Prevent loading external MathJax in Kaleido
pio.kaleido.scope.mathjax = None


def get_agieval_scores(
        model
):
    model_output_dir = '/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/agieval_solve'
    tasks = os.listdir(os.path.join(model_output_dir, model + '_checked'))
    task_scores = {}
    for file in tasks:
        with open(os.path.join(model_output_dir, model + '_checked', file), 'r') as f:
            data = json.loads(f.read())
        task_name = file.split('_out.json')[0]
        if task_name not in task_scores:
            task_scores[task_name] = -1
        accuracy = data['correct'] / (data['correct'] + data['false'])
        task_scores[task_name] = accuracy
    return task_scores


def map_agieval_scores(
        model
):
    mapping_cognitive, mapping_knowledge = get_excel_task_mapping()
    model_scores = get_agieval_scores(model)
    cognitive_scores = {}
    knowledge_scores = {}
    for task, score in model_scores.items():
        task = f'AGIEval__{task}'
        if task in mapping_cognitive:
            if mapping_cognitive[task] not in cognitive_scores:
                cognitive_scores[mapping_cognitive[task]] = []
            cognitive_scores[mapping_cognitive[task]].append(score)
        else:
            print('Unknown task:', task)
        if task in mapping_knowledge:
            if mapping_knowledge[task] not in knowledge_scores:
                knowledge_scores[mapping_knowledge[task]] = []
            knowledge_scores[mapping_knowledge[task]].append(score)
        else:
            print('Unknown task:', task)
    subtasks = [
        'aqua-rat',
        'gaokao-english',
        'logiqa-en',
        'lsat-ar',
        'lsat-lr',
        'lsat-rc',
        'math',
        'sat-en',
        'sat-math'
    ]
    for s in subtasks:
        if s == 'logiqa-en':
            task_name = 'logiqa'
        else:
            task_name = s

        score = round(model_scores[s], 6)
        # fill up to 6 decimal points with zeros
        score = f'{score:.6f}'

        print(f'{model} & AGIEval & {task_name} & {score} & Own evaluation, zero-shot prompting \\\\')
    print(f'{model} & ARC-Challenge & N/A & N/A & N/A \\\\')
    return cognitive_scores, knowledge_scores


def scores2plotly(
        model2scores
):
    groups = model2scores.keys()
    groups = [
        'gpt4',
        'llama3',
        'claude3',
        'phi3-mini-4k-it*',
        'gemma-7b-it*',
        'falcon-7b-it*',
        'falcon-40b-it*',
        'flan-t5-xxl*',
        'bloomz-3b*',
        'bloomz-560m*'
    ]
    out_cognitive = {
        'group': groups,
        'Create': [0] * len(groups),
        'Evaluate': [0] * len(groups),
        'Analyze': [0] * len(groups),
        'Apply': [0] * len(groups),
        'Understand': [0] * len(groups),
        'Remember': [0] * len(groups),
    }
    out_knowledge = {
        'group': groups,
        'Factual': [0] * len(groups),
        'Conceptual': [0] * len(groups),
        'Procedural': [0] * len(groups),
        'Metacognitive': [0] * len(groups),
    }

    colors = [
        '#000000',
        '#E69F00',
        '#56B4E9',
        '#009E73',
        '#F0E442',
        # '#0072B2',
        '#0072B2',
        '#D55E00',
        '#CC79A7',
        # '#CC79A7'
    ]

    for i, model in enumerate(groups):
        level_scores = model2scores[model]
        cognitive_scores, knowledge_scores = level_scores
        for key, scores in cognitive_scores.items():
            if isinstance(scores, list):
                scores = [s for s in scores if not np.isnan(s)]
                score = sum(scores) / len(scores)
            else:
                score = scores
            out_cognitive[key][i] = score

        for key, scores in knowledge_scores.items():
            if isinstance(scores, list):
                scores = [s for s in scores if not np.isnan(s)]
                score = sum(scores) / len(scores)
            else:
                score = scores
            out_knowledge[key][i] = score

    categories = ['Understand', 'Apply', 'Analyze', 'Evaluate', 'Create', 'Remember']

    # Reshape the DataFrame to fit Plotly Express requirements
    data = pd.DataFrame(out_cognitive)
    df_long = pd.melt(data, id_vars=['group'], value_vars=categories, var_name='category', value_name='value')

    # Explicitly set the order of the groups
    df_long['group'] = pd.Categorical(df_long['group'], categories=groups, ordered=True)

    # Create the radar chart using Plotly Express
    fig = px.line_polar(
        df_long,
        r='value',
        theta='category',
        color='group',
        line_close=True,
        template="plotly_dark",  # Change template as needed
        width=900,
        height=600
    )
    line_styles = ['solid', 'dash', 'dot']
    marker_styles = [
        'circle', 'square', 'diamond', 'cross', 'x',
        'triangle-up', 'triangle-down', 'triangle-left',
        'triangle-right', 'pentagon'
    ]

    # Update each trace with a different line style
    for i, trace in enumerate(fig.data):
        trace.update(
            line=dict(
                dash=line_styles[i % len(line_styles)],
                color=colors[i % len(colors)]
            ),
            marker=dict(symbol=marker_styles[i % len(marker_styles)], size=8),
            mode='lines+markers'
        )
    # Adjust the radial grid lines to be less intrusive or remove them
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor='black',  # Black grid lines
                gridwidth=0.5,  # Thinner grid lines for a cleaner look
                tickfont=dict(color='black', size=25),  # Black tick labels for visibility
                showline=False,
                tickangle=45,
                range=[0, 1]
            ),
            angularaxis=dict(
                linecolor='black',  # Set the color of lines to black
                linewidth=1,  # Set the width of the angular lines
                gridcolor='black',  # Set the color of angular grid lines to black
                tickfont=dict(color='black', size=25)  # Black angular tick labels for better visibility
            ),
            bgcolor='rgba(0,0,0,0)',  # Transparent polar background
        ),
        legend=dict(
            title="Model",  # Change legend title
            title_font=dict(size=30),  # Increase legend title font size
            font=dict(size=25),  # Increase legend font size
            bgcolor='rgba(0,0,0,0)',  # Set the legend background to match chart background
            bordercolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent overall background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        font=dict(color='black', size=25)  # Default font color for all text in the chart
    )

    # Save the plot to PDF
    fig.write_image("radar_chart_cognitive_more_models.pdf")

    categories = ['Conceptual', 'Procedural', 'Metacognitive', 'Factual']

    data = pd.DataFrame(out_knowledge)
    df_long = pd.melt(data, id_vars=['group'], value_vars=categories, var_name='category', value_name='value')

    # Ensure categories are correctly ordered
    df_long['category'] = pd.Categorical(df_long['category'], categories=categories, ordered=True)

    # Explicitly set the order of the groups
    df_long['group'] = pd.Categorical(df_long['group'], categories=groups, ordered=True)

    # Verify each group has all categories
    df_pivot = df_long.pivot_table(index='group', columns='category', values='value').reset_index()
    df_long = df_pivot.melt(id_vars=['group'], value_vars=categories, var_name='category', value_name='value')

    # Sort by category to maintain order
    df_long = df_long.sort_values(['group', 'category'])

    # Plot
    fig = px.line_polar(
        df_long,
        r='value',
        theta='category',
        color='group',
        line_close=True,
        template="plotly_dark",
        width=900,
        height=600
    )

    # Set line styles
    line_styles = ['solid', 'dash', 'dot']
    marker_styles = [
        'circle', 'square', 'diamond', 'cross', 'x',
        'triangle-up', 'triangle-down', 'triangle-left',
        'triangle-right', 'pentagon'
    ]
    for i, trace in enumerate(fig.data):
        trace.update(
            line=dict(
                dash=line_styles[i % len(line_styles)],
                color=colors[i % len(colors)]
            ),
            marker=dict(symbol=marker_styles[i % len(marker_styles)], size=8),
            mode='lines+markers'
        )

    # Update layout to rotate radial axis labels and adjust legend background
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor='black',
                gridwidth=0.5,
                tickfont=dict(color='black', size=25),  # Increase font size
                showline=False,
                tickangle=45,
                range=[0, 1]
            ),
            angularaxis=dict(
                linecolor='black',
                linewidth=1,
                gridcolor='black',
                tickfont=dict(color='black', size=25)  # Increase font size
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        legend=dict(
            title="Model",  # Change legend title
            title_font=dict(size=30),  # Increase legend title font size
            font=dict(size=25),  # Increase legend font size
            bgcolor='rgba(0,0,0,0)',  # Set the legend background to match chart background
            bordercolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black', size=25)  # Increase overall font size
    )

    # Save to PDF
    fig.write_image("radar_chart_knowledge_more_models.pdf")


if __name__ == '__main__':
    _models = [
        'bloomz-3b',
        'bloomz-560m',
        'falcon-7b-instruct',
        'falcon-40b-instruct',
        'flan-t5-xxl',
        'gemma-7b-it',
        'phi3-mini-4k-instruct'
    ]

    model2scores = {}
    for _model in _models:
        agieval_cognitive_scores, agieval_knowledge_scores = map_agieval_scores(_model)
        cognitive_scores, knowledge_scores = map_bbh_scores(_model)

        # merge scores
        cognitive_merged = {**cognitive_scores, **agieval_cognitive_scores}
        knowledge_merged = {**knowledge_scores, **agieval_knowledge_scores}

        print(cognitive_scores)
        print(knowledge_scores)
        print('========')

        model2scores[_model.replace('instruct', 'it') + '*'] = (cognitive_merged, knowledge_merged)

    # get_task_mapping()
    llama3_cognitive_, llama3_knowledge_ = map_bbh_scores('llama3')
    llama3_cognitive_other, llama3_knowledge_other = map_other_tasks('llama3')
    # merge them
    llama3_cognitive_merged = {}
    for cognitive in [llama3_cognitive_, llama3_cognitive_other]:
        for key, value in cognitive.items():
            if key not in llama3_cognitive_merged:
                llama3_cognitive_merged[key] = []
            llama3_cognitive_merged[key].extend(value)
    llama3_knowledge_merged = {}
    for knowledge in [llama3_knowledge_, llama3_knowledge_other]:
        for key, value in knowledge.items():
            if key not in llama3_knowledge_merged:
                llama3_knowledge_merged[key] = []
            llama3_knowledge_merged[key].extend(value)

    # llama3_cognitive_ = {**llama3_cognitive_, **llama3_cognitive_other}
    # llama3_knowledge_ = {**llama3_knowledge_, **llama3_knowledge_other}

    gpt4_cognitive_, gpt4_knowledge_ = map_bbh_scores('gpt4')
    gpt4_cognitive_other, gpt4_knowledge_other = map_other_tasks('gpt4')

    gpt4_cognitive_merged = {}
    for cognitive in [gpt4_cognitive_, gpt4_cognitive_other]:
        for key, value in cognitive.items():
            if key not in gpt4_cognitive_merged:
                gpt4_cognitive_merged[key] = []
            gpt4_cognitive_merged[key].extend(value)
    gpt4_knowledge_merged = {}
    for knowledge in [gpt4_knowledge_, gpt4_knowledge_other]:
        for key, value in knowledge.items():
            if key not in gpt4_knowledge_merged:
                gpt4_knowledge_merged[key] = []
            gpt4_knowledge_merged[key].extend(value)

    gpt4_cognitive_ = {**gpt4_cognitive_, **gpt4_cognitive_other}
    gpt4_knowledge_ = {**gpt4_knowledge_, **gpt4_knowledge_other}

    claude3_cognitive_, claude3_knowledge_ = map_bbh_scores('claude3')
    claude3_cognitive_other, claude3_knowledge_other = map_other_tasks('claude3')

    claude3_cognitive_merged = {}
    for cognitive in [claude3_cognitive_, claude3_cognitive_other]:
        for key, value in cognitive.items():
            if key not in claude3_cognitive_merged:
                claude3_cognitive_merged[key] = []
            claude3_cognitive_merged[key].extend(value)
    claude3_knowledge_merged = {}
    for knowledge in [claude3_knowledge_, claude3_knowledge_other]:
        for key, value in knowledge.items():
            if key not in claude3_knowledge_merged:
                claude3_knowledge_merged[key] = []
            claude3_knowledge_merged[key].extend(value)
    # claude3_cognitive_ = {**claude3_cognitive_, **claude3_cognitive_other}
    # claude3_knowledge_ = {**claude3_knowledge_, **claude3_knowledge_other}

    model2scores['llama3'] = (llama3_cognitive_merged, llama3_knowledge_merged)
    model2scores['gpt4'] = (gpt4_cognitive_merged, gpt4_knowledge_merged)
    model2scores['claude3'] = (claude3_cognitive_merged, claude3_knowledge_merged)

    scores2plotly(model2scores)
