
import pandas as pd
import anthropic
import os
from openai import OpenAI
from google import genai

from dotenv import load_dotenv
import json_repair as jr
import json
import numpy as np

def create_prompt_sct(domain, clinical_stem, initial_thoughts, subsequent_findings):
    
    prompt = f'''
        Your task is to assess whether the given script concordance test is realistic based on plausible, real-life context. The scenario should consist of a base description of a patient’s symptoms, an initial diagnosis, and an augmenting follow-up discovery that could change the diagnosis. It should reflect a clinician workflow in a real-world context.

        Please grade the scenario on whether it is realistic, medically accurate, and represents a diverse demographic.

        To determine if the scenario is realistic, please consider:
        Whether the realistic could naturally emerge from a real-life medical scenario or situation, rather than being a general medical example.
        Whether there is a specific context or problem motivating the scenario.
        If the patient’s issue seems like something a person would typically ask a clinician, not a theoretical issue.
        To grade this metric, please provide an integer from 1-5, following: 
        1: completely unrealistic
        2: somewhat unrealistic
        3: neither realistic nor unrealistics
        4: somewhat realistic
        5: completely realistic 
        Provide no explanation, simply the integer.

        To determine if the scenario is medically accurate, please consider:
        Whether the scenario is consistent with established medical knowledge and supported by evidence-based reasoning.
        Whether the scenario fits within the scenario’s medical domain.
        To grade this metric, please provide an integer from 1-5, following: 
        1: completely inaccurate
        2: somewhat inaccurate
        3: neither accurate nor inaccurate
        4: somewhat accurate
        5: completely accurate
        Provide no explanation, simply the integer.


        To determine if the scenario represents diverse demographics, please consider:
        Whether the scenario reflects real-world variability in populations. This includes diversity in age, sex, gender identity, race, ethnicity, socioeconomic background, disability status, and more.
        Whether the scenario reflects the cultural dynamics assigned to each group of people.
        If the scenario avoids stereotypes or bias
        To grade this metric please provide an integer from 1-5, with 1 being insufficiently diverse and completely unrepresentative of real-world populations, and 5 being sufficiently diverse and representative of real-world populations. Provide no explanation or exposition, simply the integer.

        Here are the  definitions for the data provided:
        clinical_stem: 3 to 5 sentences containing a baseline scenario about the patient’s symptoms, medical history, and important demographic information. This can be a medically complex analysis with information an examining clinician could reasonably be expected to gather about a patient from a checkup, tests, and labs.    
        initial_thoughts: An initial ordering of tests, treatment, or diagnosis based upon the initial clinical stem. This should only be a few words or small sentences.
        subsequent_finding: A follow-up finding made after the clinical stem and initial thoughts that may or may not augment the clinician’s thoughts about the patient’s treatment or diagnosis. This should force the clinician to consider the accuracy of their initial thoughts and present them with a complex medical reasoning problem.


        The entirety of the following information should be included in your analysis. The assigned medical area is {domain}. Here is the patient’s clinical stem {synth_data['clinical stem']}. Here is the initial diagnosis/treatment: {synth_data['initial_thoughts']} Here are the subsequent findings that might alter the initial treatment or diagnosis: {synth_data['subsequent_findings']}. This is all the data you should need to perform your evaluation.

        Your output should be JSON format. All numbers should be integers. Here is an example:
        "realistic": 0, "medically_accurate": 0, "diverse_demographics": 0, 
        '''
    return prompt
def gpt5_judge_api_sct(domain, row):
    synth_data = jsonify(row)
    prompt = create_sct_prompt(domain, synth_data) 
    load_dotenv()
    client = OpenAI()
    response = client.responses.create(
        model="gpt-5",
        input=prompt)

    return response.output_text
def gpt41_judge_api_sct(domain, row):
    synth_data = jsonify(row)
    prompt = create_sct_prompt(domain, synth_data) 
    load_dotenv()
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt)
    return response.output_text
    
def o3_judge_api_sct(domain, row):
    synth_data = jsonify(row)
    prompt = create_sct_prompt(domain, synth_data) 
    load_dotenv()
    client = OpenAI()
    response = client.responses.create(
        model="o3",
        input=prompt)
    return response.output_text
def cs45_judge_api_sct(domain, row):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    synth_data = jsonify(row)
    prompt = create_sct_prompt(domain, synth_data) 
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
    model="claude-sonnet-4-5",
        max_tokens=3000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    texts = [block.text for block in message.content]
    return texts[0]
def gem25p_judge_api_sct(domain, row):
    load_dotenv()
    synth_data = jsonify(row)
    prompt = create_sct_prompt(domain, synth_data) 
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt
    )

    return response.text
    
def ds_judge_api_sct(domain, row):
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    synth_data = jsonify(row)
    prompt = create_sct_prompt(domain, synth_data) 
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content
def get_model(judge_name):
# Returns a dictionary of judge functions.
    model =  {
        "gpt5": gpt5_judge_api_sct,
        "gpt41": gpt41_judge_api_sct,
        "ds": ds_judge_api_sct,
        "cs45": cs45_judge_api_sct,
        "o3": o3_judge_api_sct,
        "gem25p": gem25p_judge_api_sct
    }
    return model[judge_name]
def jsonify(table):
    data = {
        "clinical_stem": table[0],
        "initial_thoughts": table[1],
        "subsequent_finding": table[2],
    }
    return data
def sct_judge_synth(input, row, max_retries = 3):
    domain = input[0]
    model = input[1]
    for attempt in range(1, max_retries + 1):
        judge_func = get_model(model)
        raw_data = judge_func(domain, row)
        try:
            test_repair = jr.repair_json(raw_data)
            data = json.loads(test_repair)
            if not all(0 <= data[key] <= 5 for key in ["realistic", "medically_accurate", "diverse_demographics"]):
                raise ValueError("greater than 5 or less than 0")
            data_arr = np.array([
                (data['realistic']), (data['medically_accurate']), (data['diverse_demographics']), (model)
            ])
            return data_arr
        except Exception as e:
            
            if attempt >= max_retries:
                err = np.array([(0, 0, 0, "Error", model)])
                return err
                #raise ValueError("Failed :()") from e

