import json
import logging
from typing import List, Tuple

import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def parse_response(response: str):
    if not isinstance(response, str):
        return None
    response = "{" + response.split("{", 1)[-1]
    response = response.split("}", 1)[0] + "}"
    answer = None
    try:
        answer = json.loads(response)["answer"].upper()
    except Exception:
        logger.exception(f"Failed to parse response {response}")
    if answer == "FALSE":
        return False
    elif answer == "TRUE":
        return True
    else:
        # note that parsing error
        response = response.upper()
        true_index = response.find("TRUE")
        false_index = response.find("FALSE")
        if true_index > -1 and (false_index == -1 or true_index < false_index):
            return True
        elif false_index > -1 and (true_index == -1 or false_index < true_index):
            return False
        else:
            return None


def plot(
    df: pd.DataFrame,
    columns: List[str],
    model_names: List[str],
    ylabel: str,
    ylim: Tuple[int, int] | None = None,
    fname: str | None = None,
    title: str = "True/False Questions",
):
    fig, ax = plt.subplots(figsize=(6, 3))

    g = sns.barplot(
        data=df[columns],
        errorbar=("ci", 95),
        capsize=0.15,
        ax=ax,
    )
    n = len(df)
    # if title != "True/False Questions":
    #     g.set_title(f"{title} (n = {n})")
    # else:
    #     g.set_title(f"{ylabel} on (n = {n}) {title}")
    g.set_ylabel(ylabel)
    g.set_xlabel("Model")
    g.set_xticks(range(len(model_names)))
    g.set_xticklabels(model_names, rotation=45, horizontalalignment="right")
    g.set(ylim=ylim)
    g.yaxis.set_major_locator(plt.MaxNLocator(11))
    g.yaxis.grid(True)

    if fname is None:
        fname = f"/workspaces/rd-chatbot/eval/results/KG_RAG/one_hop_true_false_v2_df_{ylabel.lower()}.v2.png"

    plt.savefig(fname, bbox_inches="tight")


def metrics_llm():
    fh = logging.FileHandler("metrics.log")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    input_csv = "/workspaces/rd-chatbot/eval/results/KG_RAG/test_questions_one_hop_true_false_v2.csv"
    output_csv = input_csv.replace(".csv", "_metrics.csv")

    df = pd.read_csv(input_csv)
    for column in df.columns:
        if "response_" in column:
            label_column_name = column.replace("response_", "label_")
            df[label_column_name] = df[column].apply(parse_response)
            correct_column_name = column.replace("response_", "correct_")
            df[correct_column_name] = df[label_column_name] == df["label"]

    df.to_csv(output_csv)

    correct_columns = [column for column in df.columns if "correct_" in column]
    model_names = [column.replace("correct_", "") for column in correct_columns]

    plot(df, correct_columns, model_names, "Accuracy", (0, 1))

    time_columns = [column for column in df.columns if "time_" in column]
    for time_column in time_columns:
        df[time_column + "_seconds"] = df[time_column].apply(lambda x: pd.to_timedelta(x).total_seconds())

    time_seconds_columns = [column + "_seconds" for column in time_columns]

    plot(df, time_seconds_columns, model_names, "Time")


def metrics_translation():
    input_csv = "/workspaces/rd-chatbot/eval/results/RD/gard_corpus_translation.csv"
    output_csv = input_csv.replace(".csv", "_metrics.csv")

    df = pd.read_csv(input_csv)
    columns = []
    for column in df.columns:
        if column.startswith("translation_"):
            break
        columns.append(column)

    # for column in df.columns:
    #     if "error_" in column:
    #         assert not df[column].any()

    bleu = evaluate.load("bleu")

    column = columns[-1]

    methods = ["google", "opusmt", "mixtral8x7b", "mixtral_8x7b-instruct-v0.1-q4_0"]
    df_out = pd.DataFrame()

    for target in ["fr", "it"]:
        for method in methods:
            outputs = []
            outputs2 = []
            for i in range(len(df)):
                # calculate bleu scores using google translate as reference
                output = bleu.compute(
                    predictions=[
                        df[f"translation_{column}_{method}_en_{target}"][i],
                    ],
                    references=[
                        df[f"translation_{column}_google_en_{target}"][i],
                    ],
                )
                outputs.append(output["bleu"])
                output2 = bleu.compute(
                    predictions=[
                        df[f"back_translation_{column}_{method}_en_{target}"][i],
                    ],
                    references=[
                        df[column][i],
                    ],
                )
                outputs2.append(output2["bleu"])
            df_out[f"translation_bleu_{method}_{target}"] = outputs
            df_out[f"back_translation_bleu_{method}_{target}"] = outputs2

    df_out.to_csv(output_csv)

    bleu_columns = [
        column for column in df_out.columns if "translation_bleu_" in column and "back_translation" not in column
    ]
    bleu_columns_fr = [column for column in bleu_columns if "_fr" in column]
    cols = [
        "Google Translate",
        "OpusMT",
        # "SeamlessM4Tv2",
        "Mixtral-8x7B",
        "Mixtral-8x7B q4_0",
    ]
    plot(
        df_out,
        bleu_columns_fr,
        cols,
        "BLEU",
        (0, 1),
        "/workspaces/rd-chatbot/eval/results/RD/gard_corpus_translation_bleu_fr.png",
        title="French Translation",
    )
    bleu_columns_it = [column for column in bleu_columns if "_it" in column]
    plot(
        df_out,
        bleu_columns_it,
        cols,
        "BLEU",
        (0, 1),
        "/workspaces/rd-chatbot/eval/results/RD/gard_corpus_translation_bleu_it.png",
        title="Italian Translation",
    )

    bleu_back_columns = [column for column in df_out.columns if "back_translation_bleu_" in column]
    bleu_back_columns_fr = [column for column in bleu_back_columns if "_fr" in column]
    plot(
        df_out,
        bleu_back_columns_fr,
        cols,
        "BLEU",
        (0, 1),
        "/workspaces/rd-chatbot/eval/results/RD/gard_corpus_back_translation_bleu_fr.png",
        title="French Back Translation",
    )
    bleu_back_columns_it = [column for column in bleu_back_columns if "_it" in column]
    plot(
        df_out,
        bleu_back_columns_it,
        cols,
        "BLEU",
        (0, 1),
        "/workspaces/rd-chatbot/eval/results/RD/gard_corpus_back_translation_bleu_it.png",
        title="Italian Back Translation",
    )

    # plot time
    time_translation_columns = [column for column in df.columns if column.startswith("time_translation_")]
    time_back_translation_columns = [column for column in df.columns if column.startswith("time_back_translation_")]
    df_time = pd.concat(
        [
            df[time_translation_columns],
            df[time_back_translation_columns].rename(
                columns=lambda x: x.replace("back_translation_", "translation_")
            ),
        ]
    )
    time_columns = [column for column in df_time.columns if "time_" in column]
    time_columns = [column for column in time_columns if "seamlessm4tv2" not in column]
    for time_column in time_columns:
        df_time[time_column + "_seconds"] = df_time[time_column].apply(lambda x: pd.to_timedelta(x).total_seconds())
    time_columns_fr = [column + "_seconds" for column in time_columns if "_fr" in column]
    plot(
        df_time,
        time_columns_fr,
        cols,
        "Time (s)",
        None,
        "/workspaces/rd-chatbot/eval/results/RD/gard_corpus_translation_time_fr.png",
        title="French Translation",
    )
    time_columns_it = [column + "_seconds" for column in time_columns if "_it" in column]
    plot(
        df_time,
        time_columns_it,
        cols,
        "Time (s)",
        None,
        "/workspaces/rd-chatbot/eval/results/RD/gard_corpus_translation_time_it.png",
        title="Italian Translation",
    )


def main():
    # metrics_llm()
    metrics_translation()


if __name__ == "__main__":
    main()
