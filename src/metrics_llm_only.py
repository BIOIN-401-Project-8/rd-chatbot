
import json
import logging
from typing import List, Tuple

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


def plot(df: pd.DataFrame, columns: List[str], model_names: List[str], ylabel: str, ylim: Tuple[int, int] | None = None):
    fig, ax = plt.subplots(figsize=(12, 6))

    g = sns.barplot(
        data=df[columns],
        errorbar=("ci", 95),
        capsize=.15,
        ax=ax,
    )
    n = len(df)

    g.set_title(f"{ylabel} on (n = {n}) True/False Questions")
    g.set_ylabel("{ylabel}")
    g.set_xlabel("Model")
    g.set_xticks(range(len(model_names)))
    g.set_xticklabels(model_names, rotation=45, horizontalalignment='right')
    g.set(ylim=ylim)
    g.yaxis.set_major_locator(plt.MaxNLocator(11))
    g.yaxis.grid(True)

    plt.savefig(
        f"/workspaces/rgd-chatbot/eval/results/KG_RAG/one_hop_true_false_v2_df_{ylabel.lower()}.v2.png",
        bbox_inches="tight",
    )



def main():
    fh = logging.FileHandler("metrics.log")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    input_csv = "/workspaces/rgd-chatbot/eval/results/KG_RAG/test_questions_one_hop_true_false_v2.csv"
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


if __name__ == "__main__":
    main()
