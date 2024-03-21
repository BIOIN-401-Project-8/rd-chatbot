import time
from datetime import timedelta
import httpx
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
import logging
import pandas as pd
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from tqdm import tqdm
import re
import unicodedata
from pathlib import Path

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
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


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
    "gemma:7b-instruct-q4_0",
    "llama2:7b-chat-q4_0",
    "llama2:13b-chat-q4_0",
    "llama2:70b-chat-q4_0",
    "medllama2:7b-q4_0",
    "cniongolo/biomistral:latest",
    "mixtral:8x7b-instruct-v0.1-q4_0",
]

output_file = "/workspaces/rgd-chatbot/eval/results/KG_RAG/test_questions_one_hop_true_false_v2.csv"

df = pd.read_csv(
    "/workspaces/rgd-chatbot/eval/data/KG_RAG/test_questions_one_hop_true_false_v2.csv"
)

if Path(output_file).exists():
    df = pd.read_csv(output_file)

for model in models:
    if f"response_{slugify(model)}" in df.columns:
        continue
    try:
        logger.info(f"Loading model {model}")
        httpx.post("http://ollama:11434/api/pull", json={"name": model}, timeout=600.0)
        Settings.llm = Ollama(
            model=model,
            base_url="http://ollama:11434",
            request_timeout=60.0,
            temperature=0.0,
        )
    except:
        logger.exception(f"Failed to load model {model}")
        continue

    prompt = PromptTemplate(
'''You are an expert biomedical researcher. Please provide your answer in the following JSON format for the Question asked:
{{
"answer": "True"
}}
OR
{{
"answer": "False"
}}
{question}'''
    )

    for index, row in tqdm(df.iterrows(), total=len(df)):
        slug = slugify(model)
        error = False
        start = time.time()
        try:
            response = Settings.llm.predict(prompt, question=row["text"])
        except Exception as e:
            logger.exception(f"Failed to get response for {row['text']}")
            response = str(e)
            error = True
        end = time.time()

        df.loc[index, f"response_{slug}"] = response
        df.loc[index, f"time_{slug}"] = timedelta(seconds=end - start)
        df.loc[index, f"error_{slug}"] = error
    df.to_csv(output_file, index=False)
