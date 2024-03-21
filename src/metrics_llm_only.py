
import pandas as pd
import seaborn as sns
import json
import logging

import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
# add file handler
fh = logging.FileHandler("metrics.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

df = pd.read_csv(
    "/workspaces/rgd-chatbot/eval/results/KG_RAG/test_questions_one_hop_true_false_v2.csv"
)

def parse_response(response: str):
    try:
        if not isinstance(response, str):
            return None
        response = "{" + response.split("{", 1)[-1]
        response = response.split("}", 1)[0] + "}"
        answer = json.loads(response)["answer"]
        if answer.lower() == "false":
            return False
        elif answer.lower() == "true":
            return True
        else:
            logger.error(f"Failed to parse response {response}")
    except Exception:
        logger.exception(f"Failed to parse response {response}")
    return None


for column in df.columns:
    if "response_" in column:
        label_column_name = column.replace("response_", "label_")
        df[label_column_name] = df[column].apply(parse_response)
        correct_column_name = column.replace("response_", "correct_")
        df[correct_column_name] = df[label_column_name] == df["label"]

df.to_csv("/workspaces/rgd-chatbot/eval/results/KG_RAG/test_questions_one_hop_true_false_v2_metrics.csv")

correct_columns = [column for column in df.columns if "correct_" in column]
model_names = [column.replace("correct_", "") for column in correct_columns]
fig, ax = plt.subplots(figsize=(12, 6))

g = sns.barplot(
    data=df[correct_columns],
    errorbar=("ci", 95),
    capsize=.15,
    ax=ax,
)
n = len(df)


g.set_title(f"Accuracy on (n = {n}) True/False Questions")
g.set_ylabel("Accuracy")
g.set_xlabel("Model")
g.set_xticks(range(len(model_names)))
g.set_xticklabels(model_names, rotation=45, horizontalalignment='right')
g.set(ylim=(0, 1))

ax.tick_params(axis='x', labelrotation=45)


plt.savefig(
    "/workspaces/rgd-chatbot/eval/results/KG_RAG/one_hop_true_false_v2_df.v2.png",
    bbox_inches="tight",
)
