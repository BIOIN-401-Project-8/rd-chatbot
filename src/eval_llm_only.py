import logging
import re
import time
import unicodedata
from datetime import timedelta
from pathlib import Path

import pandas as pd
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from tqdm import tqdm

from pipelines import get_llm


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    value = value.replace(":", "_")
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


logger = logging.getLogger(__name__)
# add file handler
fh = logging.FileHandler("eval.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
# log to stdout
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

models = [
    "mistral:7b-instruct-v0.2-q4_0",
    "mistral:7b-instruct-v0.2-q8_0",
    "mistral:7b-instruct-v0.2-fp16",
    "starling-lm:7b-alpha-q4_0",
]
v = 3

output_file = f"/workspaces/rgd-chatbot/eval/results/KG_RAG/test_questions_one_hop_true_false_v{v}.csv"

df = pd.read_csv("/workspaces/rgd-chatbot/eval/data/KG_RAG/test_questions_one_hop_true_false_v2.csv")

if Path(output_file).exists():
    df = pd.read_csv(output_file)

for model_name in models:
    df_view = df
    if f"response_{slugify(model_name)}" in df.columns:
        df_view = df[df[f"error_{slugify(model_name)}"] == True]
    if len(df_view) == 0:
        continue
    try:
        logger.info(f"Loading model {model_name}")
        Settings.llm = get_llm(model_name)
    except:
        logger.exception(f"Failed to load model {model_name}")
        continue

    prompt_v2 = PromptTemplate(
        """You are an expert biomedical researcher. Please provide your answer in the following JSON format for the Question asked:
{{
"answer": "True"
}}
OR
{{
"answer": "False"
}}
{question}"""
    )

    prompt_v3 = PromptTemplate(
        """You are an expert biomedical researcher. Please provide your answer in the following format:
Answer: True
OR
Answer: False
Question: True or False {question}?
Answer: """
    )
    for index, row in tqdm(df_view.iterrows(), total=len(df_view)):
        slug = slugify(model_name)
        error = False
        start = time.time()
        try:
            response = Settings.llm.predict(prompt_v3, question=row["text"])
        except Exception as e:
            logger.exception(f"Failed to get response for {row['text']}")
            response = str(e)
            error = True
        end = time.time()

        df_view.loc[index, f"response_{slug}"] = response
        df_view.loc[index, f"time_{slug}"] = timedelta(seconds=end - start)
        df_view.loc[index, f"error_{slug}"] = error
        df.loc[df_view.index, df_view.columns] = df_view
        df.to_csv(output_file, index=False)
