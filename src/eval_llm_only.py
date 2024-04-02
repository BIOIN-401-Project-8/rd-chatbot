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
from translation import BaseTranslator, _translate, get_translator

logger = logging.getLogger(__name__)
fh = logging.FileHandler("eval.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)


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


def eval_llm():

    models = [
        "mistral:7b-instruct-v0.2-q4_0",
        "mistral:7b-instruct-v0.2-q8_0",
        "mistral:7b-instruct-v0.2-fp16",
        "starling-lm:7b-alpha-q4_0",
    ]
    v = 2

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

        prompt = PromptTemplate(
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

        for index, row in tqdm(df_view.iterrows(), total=len(df_view)):
            slug = slugify(model_name)
            error = False
            start = time.time()
            try:
                response = Settings.llm.predict(prompt, question=row["text"])
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

    return df


def eval_translation(df: pd.DataFrame):
    methods = ["google", "opusmt", "seamlessm4tv2"]
    # TODO: eval mixtral
    columns = df.columns
    for method in methods:
        translator = get_translator(method)
        for target in ["fr", "it"]:
            for column in columns:
                source = "en"
                slug = f"translation_{column}_{method}_{source}_{target}"
                eval_one_translation(df, method, translator, source, target, column, slug)
                slug = f"back_translation_{column}_{method}_{source}_{target}"
                eval_one_translation(df, method, translator, target, source, f"translation_{column}_{method}_en_{target}", slug)


def eval_one_translation(
    df: pd.DataFrame, method: str, translator: BaseTranslator, source: str, target: str, column: str, slug: str
):
    output_file = f"/workspaces/rgd-chatbot/eval/results/RD/gard_corpus_translation.csv"
    for index, row in tqdm(df.iterrows(), total=len(df)):
        error = False
        start = time.time()
        try:
            response = _translate(translator, row[column], source, target)
        except Exception as e:
            logger.exception(f"Failed to get response for {row['text']}")
            response = str(e)
            error = True
        end = time.time()
        ttime = timedelta(seconds=end - start)
        df.loc[index, f"{slug}"] = response
        df.loc[index, f"time_{slug}"] = ttime
        df.loc[index, f"error_{slug}"] = error
        df.to_csv(output_file, index=False)


def main():
    # df = eval_llm()
    df = pd.read_csv("/workspaces/rgd-chatbot/eval/data/RD/gard_corpus.csv")
    df = df.head(100)
    df = eval_translation(df)


if __name__ == "__main__":
    main()
