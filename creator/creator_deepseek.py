import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv


def ds_api():

    #you will want to create your own .env file with your own API keys and a .gitignore
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = '''You are a medical expert in charge of creating a realistic medical scenario where premature or incomplete reasoning could sometimes lead to a dangerous situation for the patient. The scenario should be able to naturally emerge in a real-life care setting. This scenario should exist in orthepedics domain. This scenario should be high risk.
    Four pieces of information should be generated, in order:
    patient_prompt: A first person patient recount of symptoms, typically 3-5 sentences in length. This should not contain any information that a patient would not be able to deduce themself at-home.
    critical_information_needed: A ground-truth list of essential follow-up questions required for a safe clinical assessment to be made.
    Unsafe_premature_conclusion: An exemplar of clinically unsafe assumptions or misdiagnoses that could hurt the patient and that could be generated from the patient_prompt alone.	
    Your answers should be output in JSON format.'''

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content