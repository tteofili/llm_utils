from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, HuggingFaceHub, OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import argparse
import arxiv
import os
from huggingface_hub.hf_api import HfFolder
from getpass import getpass


# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
HfFolder.save_token(os.getenv('HUGGINGFACEHUB_API_TOKEN'))


def generate_combined_summary(paper_texts: list, temperature: float, model_name: str, llm_service: str,
                              max_length: int = 400):
    # Concatenate the text of all papers
    combined_text = "\n".join(paper_texts)

    if llm_service == 'hf':
        prompt = "Generate a summary for the following research papers:\n{combined_text}\nThe summary should be concise and informative."
        hf_prompt = PromptTemplate.from_template(prompt)
        llm = HuggingFaceHub(repo_id=model_name, task="text-generation",
                             model_kwargs={'temperature': temperature, 'max_length': max_length,
                                           'max_new_tokens': 1024})
        chat_hf = ChatHuggingFace(llm=llm)
        chain = LLMChain(llm=chat_hf, prompt=hf_prompt)
        summary = chain.predict(combined_text=combined_text)
        #llm_chain = LLMChain(prompt=hf_prompt, llm=llm)
        #summary = print(llm_chain.run(combined_text)).content
    elif llm_service == 'openai':
        prompt = f"Generate a summary for the following research papers:\n{combined_text}\nThe summary should be concise and informative."
        # Use ChatGPT to generate a summary
        response = client.completions.create(model=model_name, prompt=prompt, temperature=temperature,
                                             max_tokens=max_length)

        # Extract the generated summary from the response
        summary = response.choices[0].text.strip()
    else:
        raise ValueError(f'unknown service type {llm_service}')

    return summary


def main(papers: list, temperature: float, model_name: str, llm_service: str):
    client = arxiv.Client()
    search_by_ids = arxiv.Search(id_list=papers)
    results = client.results(search_by_ids)
    paper_texts = [r.summary for r in results]

    # Generate a summary for the combined text of all research papers
    combined_summary = generate_combined_summary(paper_texts, temperature, model_name, llm_service)
    print("Combined Summary:")
    print(combined_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract papers summary.')
    parser.add_argument('--papers', metavar='p', type=str, nargs='+', required=True,
                        help='the paper(s) to be summarized')
    parser.add_argument('--model_name', metavar='mn', type=str, help='model name/identifier',
                        default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument('--deployment_name', metavar='dn', type=str, help='deployment name',
                        default="gpt-35-turbo")
    parser.add_argument('--temperature', metavar='tp', type=float, help='LLM temperature', default=0.01)
    parser.add_argument('--llm_service', metavar='l', type=str, help='LLM service', choices=['hf', 'openai'],
                        default='hf')

    args = parser.parse_args()
    papers = args.papers
    temperature = args.temperature
    model_name = args.model_name
    llm_service = args.llm_service
    main(papers, temperature, model_name, llm_service)
