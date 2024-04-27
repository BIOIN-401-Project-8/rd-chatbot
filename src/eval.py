#%%
import logging
import re
import time
import unicodedata
from datetime import timedelta
from itertools import product
from pathlib import Path

import pandas as pd
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from tqdm import tqdm

from pipelines import get_llm, get_pipeline
from translation import BaseTranslator, _translate, get_translator
from citation import postprocess_citation
logger = logging.getLogger(__name__)
fh = logging.FileHandler("eval.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)


def run_pipeline(llm_model_name: str, question: str):
    pipeline = get_pipeline(llm_model_name=llm_model_name)
    response = pipeline.chat(question)
    content, bibliography = postprocess_citation(response)
    if bibliography:
        content += bibliography
    return content


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
    models_llm_only = [
        "starling-lm:7b-alpha-q5_K_M",
        "llama3:8b-instruct-q5_K_M",
        "groq:llama3-8b-8192",
        "groq:llama3-70b-8192",
        "groq:mixtral-8x7b-32768",
        # "openai:gpt-4-turbo",
    ]

    output_file = f"/workspaces/rd-chatbot/eval/results/RD/test_questions.csv"

    df = pd.read_csv("/workspaces/rd-chatbot/eval/data/RD/test_questions.csv")

    if Path(output_file).exists():
        df = pd.read_csv(output_file)

    for model_name in models_llm_only:
        df_view = df
        slug = model_name
        if f"response_{slug}" in df.columns:
            df_view = df[df[f"error_{slug}"] == True]
        if len(df_view) == 0:
            continue
        llm = get_llm(llm_model_name=model_name)

        for index, row in tqdm(df_view.iterrows(), total=len(df_view)):
            error = False
            start = time.time()
            try:
                question = row["question"]
                response = llm.complete(question)
                response = response.text
            except Exception as e:
                logger.exception(f"Failed to get response for {question}")
                response = str(e)
                error = True
            end = time.time()

            df_view.loc[index, f"response_{slug}"] = response
            df_view.loc[index, f"time_{slug}"] = timedelta(seconds=end - start)
            df_view.loc[index, f"error_{slug}"] = error
            df.loc[df_view.index, df_view.columns] = df_view
            df.to_csv(output_file, index=False)

    models_rag = [
        "llama3:8b-instruct-q4_0",
    ]
    for model_name in models_rag:
        df_view = df
        slug = model_name
        slug += "_rag"
        if f"response_{slug}" in df.columns:
            df_view = df[df[f"error_{slug}"] == True]
        if len(df_view) == 0:
            continue
        llm = get_llm(llm_model_name=model_name)

        for index, row in tqdm(df_view.iterrows(), total=len(df_view)):
            slug = slug
            error = False
            start = time.time()
            try:
                question = row["question"]
                response = run_pipeline(model_name, question)
            except Exception as e:
                logger.exception(f"Failed to get response for {question}")
                response = str(e)
                error = True
            end = time.time()

            df_view.loc[index, f"response_{slug}"] = response
            df_view.loc[index, f"time_{slug}"] = timedelta(seconds=end - start)
            df_view.loc[index, f"error_{slug}"] = error
            df.loc[df_view.index, df_view.columns] = df_view
            df.to_csv(output_file, index=False)

    response_columns = [column for column in df.columns if column.startswith("response_")]
    # we want to add columns for each model
    for column in response_columns:
        slug = column.replace("response_", "")
        if f"relevant_{slug}" not in df.columns:
            df[f"relevant_{slug}"] = None
        if f"correct_{slug}" not in df.columns:
            df[f"correct_{slug}"] = None
        if f"complete_{slug}" not in df.columns:
            df[f"complete_{slug}"] = None
        if f"correctly_cited_{slug}" not in df.columns:
            df[f"correctly_cited_{slug}"] = None
    df.to_csv(output_file, index=False)
    return df


def eval_translation(df: pd.DataFrame):
    methods = ["google", "opusmt", "seamlessm4tv2", "mixtral8x7b", "mixtral_8x7b-instruct-v0.1-q4_0"]
    columns = ["text"]
    for method in methods:
        translator = get_translator(method)
        for target in ["fr", "it"]:
            for column in columns:
                source = "en"
                slug = f"translation_{column}_{method}_{source}_{target}"
                eval_one_translation(df, method, translator, source, target, column, slug)
                slug = f"back_translation_{column}_{method}_{source}_{target}"
                eval_one_translation(
                    df, method, translator, target, source, f"translation_{column}_{method}_en_{target}", slug
                )


def eval_one_translation(
    df: pd.DataFrame, method: str, translator: BaseTranslator, source: str, target: str, column: str, slug: str
):
    output_file = f"/workspaces/rd-chatbot/eval/results/RD/gard_corpus_translation.csv"
    logging.info(f"Translating {slug}")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if slug in row and pd.notna(row[slug]):
            continue
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
    df = eval_llm()
    # if Path(f"/workspaces/rd-chatbot/eval/results/RD/gard_corpus_translation.csv").exists():
    #     df = pd.read_csv("/workspaces/rd-chatbot/eval/results/RD/gard_corpus_translation.csv")
    # else:
    #     df = pd.read_csv("/workspaces/rd-chatbot/eval/data/RD/gard_corpus.csv")
    # df = df.head(100)
    # df = eval_translation(df)

#%%
if __name__ == "__main__":
    main()
