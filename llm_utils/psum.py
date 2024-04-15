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


def generate_combined_summary(paper_texts: list, temperature: float, model_name: str, llm_service: str,
                              max_length: int = 22000):
    max_length = max(max_length, len(paper_texts) * max_length)
    # Concatenate the text of all papers
    combined_text = "\n".join(paper_texts)

    if llm_service == 'hf':
        HfFolder.save_token(os.getenv('HUGGINGFACEHUB_API_TOKEN'))
        prompt = "Generate a summary for the following research papers:\n{combined_text}\nThe summary should be concise and informative."
        hf_prompt = PromptTemplate.from_template(prompt)
        # setup model locally
        llm = HuggingFaceHub(repo_id=model_name, task="text-generation",
                             model_kwargs={'temperature': temperature, 'max_length': max_length,
                                           'max_new_tokens': 1024})
        # use a chat interface
        chat_hf = ChatHuggingFace(llm=llm)
        # set up prompt
        chain = LLMChain(llm=chat_hf, prompt=hf_prompt)
        # model call
        summary = chain.predict(combined_text=combined_text)
    elif llm_service == 'openai':
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"Generate a summary for the following research papers:\n{combined_text}\nThe summary should be concise and informative."
        response = client.completions.create(model=model_name, prompt=prompt, temperature=temperature,
                                             max_tokens=max_length)

        summary = response.choices[0].text.strip()
    else:
        raise ValueError(f'unknown service type {llm_service}')

    return summary


def main(papers: list, temperature: float, model_name: str, llm_service: str, cache: str, use_summaries: bool = True):
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
            print(text)
            paper_texts.append(text)

    # Generate a summary for the combined text of all research papers
    combined_summary = generate_combined_summary(paper_texts, temperature, model_name, llm_service)
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
                        default=True)
    parser.add_argument('--llm_service', metavar='l', type=str, help='LLM service',
                        choices=['hf', 'openai'], default='hf')
    parser.add_argument('--cache', metavar='c', type=str, choices=['', 'sqlite', 'memory'], default='',
                        help='LLM prediction caching mechanism')

    args = parser.parse_args()
    papers = args.papers
    cache = args.cache
    temperature = args.temperature
    model_name = args.model_name
    llm_service = args.llm_service
    use_summaries = args.use_summaries
    main(papers, temperature, model_name, llm_service, cache, use_summaries)
