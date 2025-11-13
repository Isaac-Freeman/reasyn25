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

import creators_sct as cr
#import judges_sct as jg


# %%
#importlib.reload(jg)
importlib.reload(cr)

# %%
import os
import sys
cwd = os.getcwd()
judge_path = os.path.join(cwd, "judge")
creator_path = os.path.join(cwd, "creator")

sys.path.append(judge_path)
sys.path.append(creator_path)

# %%
input = ["psychiatry", True, True]
goob = cr.gem25p_creator_synth_sct(input)

# %%
goob
