import os
import argparse
import arxiv

from openai import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, OpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from huggingface_hub.hf_api import HfFolder


def generate_combined_summary(paper_texts: list, temperature: float, model_name: str, llm_service: str,
                              max_length: int = 400):
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


def main(papers: list, temperature: float, model_name: str, llm_service: str):
    client = arxiv.Client()
    search_by_ids = arxiv.Search(id_list=papers)
    results = client.results(search_by_ids)
    paper_texts = [r.summary for r in results]

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
    parser.add_argument('--llm_service', metavar='l', type=str, help='LLM service', choices=['hf', 'openai'],
                        default='hf')

    args = parser.parse_args()
    papers = args.papers
    temperature = args.temperature
    model_name = args.model_name
    llm_service = args.llm_service
    main(papers, temperature, model_name, llm_service)
