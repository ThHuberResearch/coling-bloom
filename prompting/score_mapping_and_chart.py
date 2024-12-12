import numpy as np
import pandas as pd

from prompting.bbh_eval.bbh_scores import get_bbh_scores_auto
import pandas as pd
import plotly.io as pio
import plotly.express as px

# Prevent loading external MathJax in Kaleido
pio.kaleido.scope.mathjax = None


def other_task_scores():
    pass


# from prompting.bbh_eval.bbh_scores import get_scores_old as get_bbh_scores


def get_excel_task_mapping() -> tuple[dict, dict]:
    df = pd.read_excel(
        '/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/EMNLP Bloom Taxonomy Classification_v4.xlsx'
    )

    mapping_cognitive = {}
    mapping_knowledge = {}
    for i, row in df.iterrows():
        task_id = f'{row["Benchmark / Dataset"]}__{row["Subtask"]}'
        mapping_cognitive[task_id] = row["Cognitive"]
        mapping_knowledge[task_id] = row["Knowledge"]
    return mapping_cognitive, mapping_knowledge


def map_other_tasks(
        model
):
    cognitive_scores = {}
    knowledge_scores = {}
    mapping_cognitive, mapping_knowledge = get_excel_task_mapping()
    # load 'Mapping' sheet
    df = pd.read_excel(
        '/home/thomas/PycharmProjects/bloom_taxonomy_emnlp/prompting/EMNLP Bloom Model Performance.xlsx',
        sheet_name='Mapping'
    )
    print(df)
    mapped_tasks = set()
    df = df[df['Model'] == model]
    for i, row in df.iterrows():
        benchmark = row['Benchmark']
        if benchmark == 'MMLU':
            continue
        subtask = row['Subtask']
        if subtask == 'logiqa':
            subtask = 'logiqa-en'
        score = row['Score']
        task_id = f'{benchmark}__{subtask}'
        cognitive = mapping_cognitive[task_id]
        if task_id in mapped_tasks:
            continue
        mapped_tasks.add(task_id)
        if cognitive not in cognitive_scores:
            cognitive_scores[cognitive] = []
        if score > 1:
            score = score / 100
        cognitive_scores[cognitive].append(score)

        knowledge = mapping_knowledge[task_id]
        if knowledge not in knowledge_scores:
            knowledge_scores[knowledge] = []
        if score > 1:
            score = score / 100
        knowledge_scores[knowledge].append(score)

    return cognitive_scores, knowledge_scores


def map_bbh_scores(
        model,
):
    mapping_cognitive, mapping_knowledge = get_excel_task_mapping()

    cognitive_scores = {}
    model_scores = get_bbh_scores_auto(model)
    import pprint
    # pprint.pprint(model_scores)
    # for task, score in model_scores.items():
    #     print(f'{model}, BBH, {task}, {score}')


    for task, score in model_scores.items():
        cognitive = mapping_cognitive[f'Big Bench Hard__{task}']
        if cognitive not in cognitive_scores:
            cognitive_scores[cognitive] = []
        cognitive_scores[cognitive].append(score)
    # for cognitive, scores in cognitive_scores.items():
    #     print(f'{cognitive}: {sum(scores) / len(scores)}')

    knowledge_scores = {}
    for task, score in model_scores.items():
        knowledge = mapping_knowledge[f'Big Bench Hard__{task}']
        if knowledge not in knowledge_scores:
            knowledge_scores[knowledge] = []
        knowledge_scores[knowledge].append(score)

    # for knowledge, scores in knowledge_scores.items():
    #     print(f'{knowledge}: {sum(scores) / len(scores)}')

    # average score
    # print(f'Average: {sum(model_scores.values()) / len(model_scores)}')

    subtasks = sorted(model_scores.keys())
    for s in subtasks:
        subtask_name = s.replace('_', '\\_')
        score = round(model_scores[s], 6)
        # fill decimals with 0 up to 6
        score = f'{score:.6f}'
        print(f'{model} & BIG-Bench Hard & {subtask_name} & {score} & Own evaluation, zero-shot prompting \\\\')

    other_tasks = [
        'DROP', 'GPQA', 'GSM8K', 'HumanEval', 'MATH', 'Winogrande'
    ]
    for o in other_tasks:
        print(f'{model} & {o} & N/A & N/A & N/A \\\\')
    return cognitive_scores, knowledge_scores


def scores2plotly(cognitive_scores, knowledge_scores):
    out_cognitive = {
        'group': ['Llama3', 'GPT-4', 'Claude3'],
        'Create': [0, 0, 0],
        'Evaluate': [0, 0, 0],
        'Analyze': [0, 0, 0],
        'Apply': [0, 0, 0],
        'Understand': [0, 0, 0],
        'Remember': [0, 0, 0],
    }
    out_knowledge = {
        'group': ['Llama3', 'GPT-4', 'Claude3'],
        'Factual': [0, 0, 0],
        'Conceptual': [0, 0, 0],
        'Procedural': [0, 0, 0],
        'Metacognitive': [0, 0, 0],
    }

    for i, cognitive in enumerate(cognitive_scores):
        for key, score in cognitive.items():
            # filter out all nan values
            score = [s for s in score if not np.isnan(s)]
            if len(score) > 0:
                out_cognitive[key][i] = sum(score) / len(score)

    for i, knowledge in enumerate(knowledge_scores):
        for key, score in knowledge.items():
            # filter out all nan values
            score = [s for s in score if not np.isnan(s)]
            if len(score) > 0:
                out_knowledge[key][i] = sum(score) / len(score)

    categories = ['Understand', 'Apply', 'Analyze', 'Evaluate', 'Create', 'Remember']

    # Reshape the DataFrame to fit Plotly Express requirements
    data = pd.DataFrame(out_cognitive)
    df_long = pd.melt(data, id_vars=['group'], value_vars=categories, var_name='category', value_name='value')

    # Explicitly set the order of the groups
    df_long['group'] = pd.Categorical(df_long['group'], categories=['Llama3', 'GPT-4', 'Claude3'], ordered=True)

    # Create the radar chart using Plotly Express
    fig = px.line_polar(
        df_long,
        r='value',
        theta='category',
        color='group',
        line_close=True,
        template="plotly_dark",  # Change template as needed
    )
    line_styles = ['solid', 'dash', 'dot']

    # Update each trace with a different line style
    for i, trace in enumerate(fig.data):
        trace.update(line=dict(dash=line_styles[i % len(line_styles)]))

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
    fig.write_image("radar_chart_cognitive.pdf")

    categories = ['Conceptual', 'Procedural', 'Metacognitive', 'Factual']

    data = pd.DataFrame(out_knowledge)
    df_long = pd.melt(data, id_vars=['group'], value_vars=categories, var_name='category', value_name='value')

    # Ensure categories are correctly ordered
    df_long['category'] = pd.Categorical(df_long['category'], categories=categories, ordered=True)

    # Explicitly set the order of the groups
    df_long['group'] = pd.Categorical(df_long['group'], categories=['Llama3', 'GPT-4', 'Claude3'], ordered=True)

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
    )

    # Set line styles
    line_styles = ['solid', 'dash', 'dot']
    for i, trace in enumerate(fig.data):
        trace.update(line=dict(dash=line_styles[i % len(line_styles)]))

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
    fig.write_image("radar_chart_knowledge.pdf")


if __name__ == '__main__':
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

    scores2plotly(
        [llama3_cognitive_merged, gpt4_cognitive_merged, claude3_cognitive_merged],
        [llama3_knowledge_merged, gpt4_knowledge_merged, claude3_knowledge_merged]
    )
