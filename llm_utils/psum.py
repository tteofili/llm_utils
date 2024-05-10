import os
import argparse

import arxiv
from arxiv2text import arxiv_to_text

from openai import OpenAI
import langchain
from langchain.cache import InMemoryCache, SQLiteCache
from langchain import PromptTemplate, HuggingFaceHub, OpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from huggingface_hub.hf_api import HfFolder

DEFAULT_PROMPT_TEMPLATE = (
    "Generate a summary for the following {num_papers} research papers (every paper starts with a {paper_header} line):"
    "\n{combined_text}\nThe summary should be informative and provide implementation details."
    "\nAlso suggest how to possibly combine methods from the {num_papers} papers, with implementation details.")


def generate_combined_summary(paper_texts: list, temperature: float, model_name: str, llm_service: str,
                              prompt_template: str = DEFAULT_PROMPT_TEMPLATE, max_length: int = 22000):
    max_length = max(max_length, len(paper_texts) * max_length)

    # Combine the text of the papers
    paper_header = "**PAPER**"
    combined_text = join_papers(paper_texts, paper_header)

    if llm_service == 'hf':
        HfFolder.save_token(os.getenv('HUGGINGFACEHUB_API_TOKEN'))
        hf_prompt = PromptTemplate.from_template(prompt_template)
        # setup model locally
        llm = HuggingFaceHub(repo_id=model_name, task="text-generation",
                             model_kwargs={'temperature': temperature, 'max_length': max_length,
                                           'max_new_tokens': 1024})
        # use a chat interface
        chat_hf = ChatHuggingFace(llm=llm)
        # set up prompt
        chain = LLMChain(llm=chat_hf, prompt=hf_prompt)
        # model call
        summary = chain.predict(combined_text=combined_text, num_papers=len(paper_texts), paper_header=paper_header)
    elif llm_service == 'openai':
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt_template = f"Generate a summary for the following research papers:\n{combined_text}\nThe summary should be concise and informative."
        response = client.completions.create(model=model_name, prompt=prompt_template, temperature=temperature,
                                             max_tokens=max_length)

        summary = response.choices[0].text.strip()
    else:
        raise ValueError(f'unknown service type {llm_service}')

    return summary


def join_papers(paper_texts, paper_header, max_input_length: int = 32000):
    filtered_texts = []
    for paper_text in paper_texts:
        # filter references out
        filtered_text = paper_text[:paper_text.find("REFERENCES\n[1]")]
        # filter out last part of the paper
        filtered_text = filtered_text[:int(max_input_length / len(paper_texts))]
        filtered_texts.append(filtered_text)
    return f"\n\n{paper_header}\n\n".join(filtered_texts)


def main(papers: list, temperature: float, model_name: str, llm_service: str, cache: str, use_summaries: bool = True,
         prompt_template: str = DEFAULT_PROMPT_TEMPLATE):
    if cache == "memory":
        langchain.llm_cache = InMemoryCache()
    elif cache == "sqlite":
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    client = arxiv.Client()
    search_by_ids = arxiv.Search(id_list=papers)
    results = client.results(search_by_ids)
    if use_summaries:
        paper_texts = [r.summary for r in results]
    else:
        paper_texts = []
        for r in results:
            text = arxiv_to_text(r.pdf_url)
            paper_texts.append(text)

    # Generate a summary for the combined text of all research papers
    combined_summary = generate_combined_summary(paper_texts, temperature, model_name, llm_service, prompt_template)
    print("Combined Summary:")
    print(combined_summary.split('[/INST]')[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract papers summary.')
    parser.add_argument('--papers', metavar='p', type=str, nargs='+', required=True,
                        help='the paper(s) to be summarized')
    parser.add_argument('--model_name', metavar='mn', type=str, help='model name/identifier',
                        default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument('--deployment_name', metavar='dn', type=str, help='deployment name',
                        default="gpt-35-turbo")
    parser.add_argument('--temperature', metavar='tp', type=float, help='LLM temperature', default=0.01)
    parser.add_argument('--use_summaries', metavar='us', type=bool, help='whether to use abstracts or not',
                        default=False)
    parser.add_argument('--llm_service', metavar='l', type=str, help='LLM service',
                        choices=['hf', 'openai'], default='hf')
    parser.add_argument('--cache', metavar='c', type=str, choices=['', 'sqlite', 'memory'], default='',
                        help='LLM prediction caching mechanism')
    parser.add_argument('--prompt_template', metavar='pt', type=str, default=DEFAULT_PROMPT_TEMPLATE,
                        help='LLM prompt template')

    args = parser.parse_args()
    papers = args.papers
    cache = args.cache
    temperature = args.temperature
    model_name = args.model_name
    llm_service = args.llm_service
    use_summaries = args.use_summaries
    prompt_template = args.prompt_template
    main(papers, temperature, model_name, llm_service, cache, use_summaries, prompt_template)
