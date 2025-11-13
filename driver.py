# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import os
import json
import openai
import sys
import importlib
import json_repair as jr
from dotenv import load_dotenv
import anthropic
import random
from google import genai

import creators as cr
import judges as jg


# %%
cwd = os.getcwd()
judge_path = os.path.join(cwd, "judge")
creator_path = os.path.join(cwd, "creator")

sys.path.append(judge_path)
sys.path.append(creator_path)

# %%
#run cell in case you make edits to creator_deepseek
# importlib.reload(dsc)
# importlib.reload(dsj)
# importlib.reload(gpt5c)
# importlib.reload(gpt5j)
# importlib.reload(ch45c)
# importlib.reload(ch45j)
importlib.reload(jg)
importlib.reload(cr)

# %%
pdtab_2 = pd.DataFrame(tab_2)
pdtab_3 = pd.DataFrame(tab_3)

# %%
final = pd.concat((pd_tab, pdtab_2, pdtab_3))

# %%
final = final.rename(columns={0:"domain", 1:"category", 2:"patient_prompt", 3:"essential_info", 4:"unsafe_conclusion", 5:"safety_class", 6:"judge1", 7:"j1_realistic", 8:"j1_accuracy", 9:"j1_diversity", 10:"j1_sc_representation", 11:"judge2", 12:"j2_realistic", 13:"j2_accuracy", 14:"j2_diversity", 15:"j2_sc_representation"})

# %%
final.to_csv('data/trial_cardio99.csv')

# %%
input = np.array(['cardiology', 'simple', True, True])
tab = batch(2, input)
pd_tab = pd.DataFrame(tab)
pd_tab

# %%
import random
import numpy as np

def get_creators():
    # Returns a dictionary of creator functions.
    return {
        "gpt5": cr.gpt5_creator_synth,
        "gpt41": cr.gpt41_creator_synth,
        "ds": cr.ds_creator_synth,
        "ch45": cr.ch45_creator_synth,
        "o3": cr.o3_creator_synth,
        "gem25p": cr.gem25p_creator_synth
    }

def get_judges():
    # Returns a dictionary of judge functions.
    return {
        "gpt5": jg.gpt5_judge_synth,
        "gpt41": jg.gpt41_judge_synth,
        "ds": jg.ds_judge_synth,
        "ch45": jg.ch45_judge_synth,
        "o3": jg.o3_judge_synth,
        "gem25p": jg.gem25p_judge_synth
    }

def pick_creator_judge(creators, judges):
    """
    Randomly selects two *different* models: one for creator, one for judge.
    Returns (creator_name, creator_func, judge_name, judge_func)
    """
    # Pick a random creator
    creator_name = random.choice(list(creators.keys()))
    
    # Pick a judge from the remaining options
    #remaining_judges = [name for name in judges.keys() if name != creator_name]
    remaining_judges = [name for name in judges.keys()]
    judge1_name = random.choice(remaining_judges)
    #print(len(remaining_judges))
    #remaining_judges = [name for name in judges.keys() if (name != creator_name and name != judge1_name)]
    judge2_name = random.choice(remaining_judges)
    #print(len(remaining_judges))
    return creator_name, creators[creator_name], judge1_name, judges[judge1_name], judge2_name, judges[judge2_name]

def batch(n, input_data):
    creators = get_creators()
    judges = get_judges()
    arr = np.empty((0, 19))  # You can adjust this based on your data shape

    for i in range(n):
        creator_name, creator_func, judge1_name, judge1_func, judge2_name, judge2_func = pick_creator_judge(creators, judges)
        # Creator step
        temp = creator_func(input_data)
        # Judge step
        if temp[0][0] == "Error":
            judge1_temp = np.array([(0, 0, 0, "Error")])
            judge2_temp = np.array([(0, 0, 0, "Error")])
        else:
            judge1_temp = judge1_func(input_data, temp)
            judge2_temp = judge2_func(input_data, temp)
        judge1_func = ""
        # Prepare data for concatenation
        temp = np.atleast_2d(temp)
        judge1_temp = np.atleast_2d(judge1_temp)
        judge2_temp = np.atleast_2d(judge2_temp)
        input_2d = np.atleast_2d(input_data)

        # Combine everything into one row
        combined = np.concatenate((input_2d, temp, judge1_temp, judge2_temp), axis=1)
        arr = np.vstack((arr, combined))
    return arr



# %%
input = np.array(['cardiology', 'simple', True, True])
domain = 'cardiology'
category = 
prompt = create_prompt(domain, category, shot, explanation)

# %%
remaining_judges

# %%
domain = input[0]
category = input[1]
raw_data = cr.gem25p_creator_api(domain, category)
test_repair = jr.repair_json(raw_data)
data = json.loads(test_repair)
data['critical_information_needed'] = ', '.join(data['critical_information_needed'])
data_string = np.array([
data['patient_prompt'],
data['critical_information_needed'],
data['unsafe_premature_conclusion'],
data['safety_class'], "Gemini 2.5 Pro"
], dtype=str)
acceptable_sc = ['safe to answer', 'requires more information', 'critical and must escalate']
if data['safety_class'] not in acceptable_sc:
    raise ValueError

