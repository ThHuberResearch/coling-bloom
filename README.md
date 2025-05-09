Repo for our paper 'LLMs meet Bloom's Taxonomy: A Cognitive View on Large Language Model Evaluations'

Contains the outputs for the models on the benchmarks, as well as their assessment of the cognitive dimensions and
knowledge types.

# Abstract

Current evaluation approaches for Large Language Models (LLMs) lack a structured approach that reflects the underlying
cognitive abilities required for solving the tasks.
This hinders a thorough understanding of the current level of LLM capabilities. For instance, it is widely accepted that
LLMs perform well in terms of grammar, but it is unclear in what specific cognitive areas they excel or struggle.
This paper introduces a novel perspective on the evaluation of LLMs that leverages a hierarchical classification of
tasks. Specifically, we explore the most widely used benchmarks for LLMs to systematically identify how well these
existing evaluation methods cover the levels of Bloom's Taxonomy, a hierarchical framework for categorizing cognitive
skills. This comprehensive analysis allows us to identify strengths and weaknesses in current LLM assessment strategies
in terms of cognitive abilities and suggest directions for both future benchmark development as well as highlight
potential avenues for LLM research. Our findings reveal that LLMs generally perform better on the lower end of Bloom's
Taxonomy. Additionally, we find that there are significant gaps in the coverage of cognitive skills in the most commonly
used benchmarks.

# BBH and AGIEval evaluation answers

The model outputs, with the automated evaluation, can be found in `prompting/answer_checking`.

# Cognitive dimensions and knowledge types classifications

The labels assigned by the models can be found in `prompting/classify`.

# Citation

If you use this code or the data in your research, please cite [our paper](https://aclanthology.org/2025.coling-main.350/):

```bibtex
@inproceedings{huber-niklaus-2025-llms,
    title = "{LLM}s meet Bloom`s Taxonomy: A Cognitive View on Large Language Model Evaluations",
    author = "Huber, Thomas  and
      Niklaus, Christina",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.350/",
    pages = "5211--5246",
    abstract = "Current evaluation approaches for Large Language Models (LLMs) lack a structured approach that reflects the underlying cognitive abilities required for solving the tasks. This hinders a thorough understanding of the current level of LLM capabilities. For instance, it is widely accepted that LLMs perform well in terms of grammar, but it is unclear in what specific cognitive areas they excel or struggle in. This paper introduces a novel perspective on the evaluation of LLMs that leverages a hierarchical classification of tasks. Specifically, we explore the most widely used benchmarks for LLMs to systematically identify how well these existing evaluation methods cover the levels of Bloom`s Taxonomy, a hierarchical framework for categorizing cognitive skills. This comprehensive analysis allows us to identify strengths and weaknesses in current LLM assessment strategies in terms of cognitive abilities and suggest directions for both future benchmark development as well as highlight potential avenues for LLM research. Our findings reveal that LLMs generally perform better on the lower end of Bloom`s Taxonomy. Additionally, we find that there are significant gaps in the coverage of cognitive skills in the most commonly used benchmarks."
}