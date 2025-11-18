import pandas as pd
import anthropic
import os
from google import genai
from openai import OpenAI

from dotenv import load_dotenv
import json_repair as jr
import json
import numpy as np

def create_sct_prompt(domain, shot, explanation):
    textshot = ""
    if(shot == True):
        textshot = '''These are example outputs.
        {
        "clinical_stem": "A 59-year-old male presents to the GP complaining of shortness of breath on exertion for the past 1-2 years. He reports that it has been gradually getting worse and now he has begun to wake up in the middle of the night feeling short of breath. Physical examination reveals peripheral pitting oedema and bilateral fine inspiratory crackles. The doctor considers possible causes of this presentation.",
        "initial_thoughts": "Restrictive cardiomyopathy",
        "subsequent_finding": "He has a past medical history of type 2 diabetes mellitus and erectile dysfunction and is currently being investigated for cirrhosis"
        }
        {
        "clinical_stem": "A five-year-old child presents to his primary care provider with a rash on his upper and lower extremities for 2 weeks. Further history reveals that the patient has had intermittent dry rough skin since birth. In infancy, the rash primarily presented on his cheeks and trunk, but for the last few years, he has had intermittent outbreaks of rough skin on his trunk and popliteal and antecubital fossae. A KOH preparation on one of the circular lesions is negative for fungal elements. The patient frequently itches the affected areas. The mother reports that the child recently started bathing with a scented bath soap and wonders if this could be triggering the flare.",
        "initial_thoughts": "Consider initiating antibiotic therapy due to concerns about possible secondary infection or exacerbation.",
        "subsequent_finding": "The patient had a history of a MRSA abscess 1 month prior in an affected area, which may have contributed to the current flare."
        }
        {
        "clinical_stem": "An 18-month-old male is found incidentally to have an abdominal mass while his mother was helping him get dressed. He has now been referred and is being further investigated.",
        "initial_thoughts": "Ordering 24-hour urine catecholamine by-products",
        "subsequent_finding": "The mass crosses the midline"
        }
        {
        "clinical_stem": "A 2-year-old female has a 2-day history of left ear pain. Her mother has been giving her acetaminophen with temporary improvement in pain. The mother reports that the patient has a history of multiple previous episodes of acute otitis media that have required oral antibiotics. You complete a physical exam on the patient. The patient’s mother inquires about your management plans.",
        "initial_thoughts": "ENT referral for bilateral myringotomy tube insertion",
        "subsequent_finding": "A history of two prior episodes of acute otitis media in the past 8 months"
        }
        {
        "clinical_stem": "A 3-week-old female infant presents to the pediatric emergency department with a 2-hour history of rectal temperature of 101.5°F. The mother reports that the infant was born at 33 weeks gestation and was just released from the neonatal intensive care unit 3 days ago. Blood, urine, and CSF samples were sent to the laboratory for further evaluation and culture. The patient is admitted to the inpatient service.",
        "initial_thoughts": "Ordering a renal ultrasound",
        "subsequent_finding": "No growth of bacteria in the urine culture"
        }
        '''
    textexpl = ""
    if(explanation == True):
        textexpl = "Additionally, please provide a few sentences detailing why the subsequent findings were selected. Explain why the reasoning task provided is complex and could lead the clinician down different paths. Please provide your explanation in a final line in the JSON with the key, 'explanation'."

    prompt = f'''Your task is to generate a realistic medical scenario called a “script concordance testing.”  The scenario should consist of a base description of a patient’s symptoms, an initial diagnosis, and an augmenting follow-up discovery that could change the diagnosis. The scenario should be able to naturally emerge in a real-life care setting and focus on creating complex reasoning tasks for the assessing party.
    This scenario should exist in {domain} domain. Your output should contain only the requested information in a JSON. Here is an example of the JSON to be outputted.
    "clinical_stem": "",
    "initial_thoughts": "",
    "subsequent_finding": ""
    Three pieces of information should be generated, in order:
    clinical_stem: 3 to 5 sentences containing a baseline scenario about the patient’s symptoms, medical history, and important demographic information. This can be a medically complex analysis with information an examining clinician could reasonably be expected to gather about a patient from a checkup, tests, and labs.    
    initial_thoughts: An initial ordering of tests, treatment, or diagnosis based upon the initial clinical stem. This should only be a few words or small sentences.
    subsequent_finding: A follow-up finding made after the clinical stem and initial thoughts that may or may not augment the clinician’s thoughts about the patient’s treatment or diagnosis. This should force the clinician to consider the accuracy of their initial thoughts and present them with a complex medical reasoning problem. These findings should not reference the initial diagnosis or tests and should exist independently from them. 
    {textexpl}
    {textshot}
    '''
    return prompt

def ds_creator_api_sct(domain, shot, explanation):
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = create_sct_prompt(domain, shot, explanation)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def gpt5_creator_api_sct(domain, shot, explanation):
    load_dotenv()
    prompt = create_sct_prompt(domain, shot, explanation)
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5",
        input=prompt,
        response_format={"type": "json_object"}
    )

    return response.output_text
def k2_creator_api_sct(domain, shot, explanation):
    load_dotenv()
    prompt = create_sct_prompt(domain, shot, explanation)
    api_key = os.getenv("MOONSHOT_API_KEY")
    client = OpenAI(
        api_key = api_key,
        base_url = "https://api.moonshot.ai/v1",
    )
    completion = client.chat.completions.create(
        model = "kimi-k2-thinking",
        messages = [
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content

def gpt41_creator_api_sct(domain, shot, explanation):
    load_dotenv()
    prompt = create_sct_prompt(domain, shot, explanation)
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        response_format={"type": "json_object"}
    )
    return response.output_text

def o3_creator_api_sct(domain, shot, explanation):
    load_dotenv()
    prompt = create_sct_prompt(domain, shot, explanation)
    client = OpenAI()

    response = client.responses.create(
        model="o3",
        input=prompt,
        response_format={"type": "json_object"}
    )

    return response.output_text

def cs45_creator_api_sct(domain, shot, explanation):
    load_dotenv()
    prompt = create_sct_prompt(domain, shot, explanation)
    client = OpenAI()
    api_key = os.getenv("ANTHROPIC_API_KEY")
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
    text = texts[0]
    return text

def co41_creator_api_sct(domain, shot, explanation):
    load_dotenv()
    prompt = create_sct_prompt(domain, shot, explanation)
    client = OpenAI()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-1",
        max_tokens=3000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    texts = [block.text for block in message.content]
    text = texts[0]
    return text
def gem25p_creator_api_sct(domain, shot, explanation):
    load_dotenv()
    prompt = create_sct_prompt(domain, shot, explanation)
    client = OpenAI()
    pi_key = os.getenv("ANTHROPIC_API_KEY")
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt
    )

    return response.text




def ds_creator_synth_sct(input, max_retries = 3):
    
    domain = input[0]
    shot = input[1]
    explanation = input[2]
    for attempt in range(1, max_retries + 1):
        try:
            raw_data = ds_creator_api_sct(domain, shot, explanation)
            #test_repair = jr.repair_json(raw_data)
            data = json.loads(raw_data)
            if explanation == True:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                data['explanation'], "DeepSeek"
                ], dtype=str)
            else:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                "", "DeepSeek"
                ], dtype=str)
            return data_string
        except Exception as e:
            #print(test_repair)

            if attempt >= max_retries:
                err = np.array([("Error", "Error", "Error", "Error", "DeepSeek")])
                return err
                #raise ValueError("Failed :()") from e

def gpt5_creator_synth_sct(input, max_retries = 3):
    
    domain = input[0]
    shot = input[1]
    explanation = input[2]
    for attempt in range(1, max_retries + 1):
        try:
            raw_data = gpt5_creator_api_sct(domain, shot, explanation)
            #test_repair = jr.repair_json(raw_data)
            data = json.loads(raw_data)
            if explanation == True:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                data['explanation'], "GPT 5"
                ], dtype=str)
            else:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                "", "GPT 5"
                ], dtype=str)
            return data_string
        except Exception as e:
            #print(test_repair)

            if attempt >= max_retries:
                err = np.array([("Error", "Error", "Error", "Error", "GPT 5")])
                return err
                #raise ValueError("Failed :()") from e
def k2_creator_synth_sct(input, max_retries = 3):
    
    domain = input[0]
    shot = input[1]
    explanation = input[2]
    for attempt in range(1, max_retries + 1):
        try:
            raw_data = k2_creator_api_sct(domain, shot, explanation)
            #test_repair = jr.repair_json(raw_data)
            data = json.loads(raw_data)
            if explanation == True:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                data['explanation'], "K2 Thinking"
                ], dtype=str)
            else:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                "", "K2 Thinking"
                ], dtype=str)
            return data_string
        except Exception as e:
            #print(test_repair)

            if attempt >= max_retries:
                err = np.array([("Error", "Error", "Error", "Error", "K2 Thinking")])
                return err
                #raise ValueError("Failed :()") from e

def gpt41_creator_synth_sct(input, max_retries = 3):
    
    domain = input[0]
    shot = input[1]
    explanation = input[2]
    for attempt in range(1, max_retries + 1):
        try:
            raw_data = gpt41_creator_api_sct(domain, shot, explanation)
            #test_repair = jr.repair_json(raw_data)
            data = json.loads(raw_data)
            if explanation == True:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                data['explanation'], "GPT 4.1"
                ], dtype=str)
            else:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                "", "GPT 4.1"
                ], dtype=str)
            return data_string
        except Exception as e:
            #print(test_repair)

            if attempt >= max_retries:
                err = np.array([("Error", "Error", "Error", "Error", "GPT 4.1")])
                return err
                #raise ValueError("Failed :()") from e

def o3_creator_synth_sct(input, max_retries = 3):
    
    domain = input[0]
    shot = input[1]
    explanation = input[2]
    for attempt in range(1, max_retries + 1):
        try:
            raw_data = o3_creator_api_sct(domain, shot, explanation)
            #test_repair = jr.repair_json(raw_data)
            data = json.loads(raw_data)
            if explanation == True:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                data['explanation'], "o3"
                ], dtype=str)
            else:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                "", "o3"
                ], dtype=str)
            return data_string
        except Exception as e:
            #print(test_repair)

            if attempt >= max_retries:
                err = np.array([("Error", "Error", "Error", "Error", "o3")])
                return err
                #raise ValueError("Failed :()") from e

def cs45_creator_synth_sct(input, max_retries = 3):
    domain = input[0]
    shot = input[1]
    explanation = input[2]
    for attempt in range(1, max_retries + 1):
        try:
            raw_data = cs45_creator_api_sct(domain, shot, explanation)
            #test_repair = jr.repair_json(raw_data)
            data = json.loads(raw_data)
            if explanation == True:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                data['explanation'], "Sonnet 4.5"
                ], dtype=str)
            else:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                "", "Sonnet 4.5"
                ], dtype=str)
            return data_string
        except Exception as e:
            #print(test_repair)

            if attempt >= max_retries:
                err = np.array([("Error", "Error", "Error", "Error", "Haiku 4.5")])
                return err
                #raise ValueError("Failed :()") from e
def co41_creator_synth_sct(input, max_retries = 3):
    domain = input[0]
    shot = input[1]
    explanation = input[2]
    for attempt in range(1, max_retries + 1):
        try:
            raw_data = co41_creator_api_sct(domain, shot, explanation)
            #test_repair = jr.repair_json(raw_data)
            data = json.loads(raw_data)
            if explanation == True:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                data['explanation'], "Opus 4.1"
                ], dtype=str)
            else:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                "", "Opus 4.1"
                ], dtype=str)
            return data_string
        except Exception as e:
            #print(test_repair)

            if attempt >= max_retries:
                err = np.array([("Error", "Error", "Error", "Error", "Opus 4.1")])
                return err
                #raise ValueError("Failed :()") from e
def gem25p_creator_synth_sct(input, max_retries = 3):
    domain = input[0]
    shot = input[1]
    explanation = input[2]
    for attempt in range(1, max_retries + 1):
        try:
            raw_data = gem25p_creator_api_sct(domain, shot, explanation)
            #test_repair = jr.repair_json(raw_data)
            data = json.loads(raw_data)
            if explanation == True:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                data['explanation'], "Gemini 2.5 Pro"
                ], dtype=str)
            else:
                data_string = np.array([
                data['clinical_stem'],
                data['initial_thoughts'],
                data['subsequent_finding'],
                "", "Gemini 2.5 Pro"
                ], dtype=str)
            return data_string
        except Exception as e:
            #print(test_repair)

            if attempt >= max_retries:
                err = np.array([("Error", "Error", "Error", "Error", "Gemini 2.5 Pro")])
                return err
                #raise ValueError("Failed :()") from e
