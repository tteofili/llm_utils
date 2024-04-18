# llm_utils

a set of utils based on LLMs

## paper summary

with `psum` one can get quick summaries from [arXiv](https://arxiv.org/) papers.
pass one paper identifier on the command line to get the summary of such a paper or pass multiple such identifiers to get a summary of multiple papers.
```
$> python llm_utils/psum.py --papers 2401.14887 2401.14887v3

Combined Summary:
 This research analyzes the impact of Information Retrieval (IR) components on Retrieval-Augmented Generation (RAG) systems, which enhance traditional Large Language Models (LLMs) by incorporating external data. The study focuses on the optimal characteristics of a retriever for effective prompt formulation, specifically the type of documents to retrieve. The findings reveal that including irrelevant documents can increase performance by over 30% in accuracy, contradicting initial assumptions. This research highlights the need for specialized strategies to integrate retrieval with language generation models, paving the way for future research in this field.
```

it works both on OpenAI and HF APIs, HF is the default.
by default it uses a [Mixtral-7B instructed model](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).
before running the script, make sure to export your HF token, e.g. `export HUGGINGFACEHUB_API_TOKEN=123token456`.
