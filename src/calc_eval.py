# %%
import pandas as pd

one_hop_true_false_v2_df = pd.read_csv(
    "/workspaces/rgd-chatbot/eval/results/KG_RAG/one_hop_true_false_v2_df.csv"
)
# %%
import seaborn as sns

n = len(one_hop_true_false_v2_df)

# plot bar plot with confidence intervals on the accuracy of the llm and rag

one_hop_true_false_v2_df["llm_acc"] = one_hop_true_false_v2_df["label"] == one_hop_true_false_v2_df["llm"]
one_hop_true_false_v2_df["rag_acc"] = one_hop_true_false_v2_df["label"] == one_hop_true_false_v2_df["rag_ans"]


import re
one_hop_true_false_v2_df["llm_has_source"] = one_hop_true_false_v2_df["llm_predict"].str.contains(r"Source \d+", flags=re.I)
one_hop_true_false_v2_df["rag_has_source"] = one_hop_true_false_v2_df["rag_predict"].str.contains(r"Source \d+", flags=re.I)

g = sns.barplot(
    data=one_hop_true_false_v2_df[["llm_acc", "rag_acc"]].head(n),
    errorbar=("ci", 95),
    capsize=.15,
    # ylim=(0, 1),
)
g.set_title(f"Accuracy of LLM vs RAG on (n = {n}) True/False Questions")
g.set_ylabel("Accuracy")
g.set_xlabel("Model")
g.set_xticklabels(["LLM", "RAG"])
g.set(ylim=(0, 1))
# save to file

import matplotlib.pyplot as plt

plt.savefig("/workspaces/rgd-chatbot/eval/results/KG_RAG/one_hop_true_false_v2_df.png")

# %%
# make a plot of the time

one_hop_true_false_v2_df["llm_time_s"] = one_hop_true_false_v2_df["llm_time"].apply(lambda x: pd.to_timedelta(x).total_seconds())
one_hop_true_false_v2_df["rag_time_s"] = one_hop_true_false_v2_df["rag_time"].apply(lambda x: pd.to_timedelta(x).total_seconds())

g = sns.barplot(
    data=one_hop_true_false_v2_df[["llm_time_s", "rag_time_s"]].head(n),
    errorbar=("ci", 95),
    capsize=.15,
)
g.set_title(f"Time to predict LLM vs RAG on (n = {n}) True/False Questions")
g.set_ylabel("Time (s)")
g.set_xlabel("Model")
g.set_xticklabels(["LLM", "RAG"])
# save to file
plt.savefig("/workspaces/rgd-chatbot/eval/results/KG_RAG/one_hop_true_false_v2_df_time.png")
# %%
# make a plot of has source

g = sns.barplot(
    data=one_hop_true_false_v2_df[["llm_has_source", "rag_has_source"]].head(n),
    errorbar=("ci", 95),
    capsize=.15,
)
g.set_title(f"Has Source in LLM vs RAG on (n = {n}) True/False Questions")
g.set_ylabel("Has Source")
g.set_xlabel("Model")
g.set_xticklabels(["LLM", "RAG"])
# save to file
plt.savefig("/workspaces/rgd-chatbot/eval/results/KG_RAG/one_hop_true_false_v2_df_has_source.png")

#%%
# side by side print
one_hop_true_false_v2_df.head(n)
# %%
# TODO: remove code duplication, cleanup script, make reusable

# %%

# %%
# %%
# calc accuracy std and mean time and std
def calc_metrics(df):
    llm_acc = df["llm_acc"].mean()
    llm_acc_std = df["llm_acc"].std()
    llm_acc_ci = 1.96 * llm_acc_std / (len(df) ** 0.5)
    llm_acc_ci = (llm_acc - llm_acc_ci, llm_acc + llm_acc_ci)
    rag_acc = df["rag_acc"].mean()
    rag_acc_std = df["rag_acc"].std()
    rag_acc_ci = 1.96 * rag_acc_std / (len(df) ** 0.5)
    rag_acc_ci = (rag_acc - rag_acc_ci, rag_acc + rag_acc_ci)
    llm_time = df["llm_time_s"].mean()
    llm_time_sum = df["llm_time_s"].sum()
    llm_time_std = df["llm_time_s"].std()
    rag_time = df["rag_time_s"].mean()
    rag_time_sum = df["rag_time_s"].sum()
    rag_time_std = df["rag_time_s"].std()
    llm_has_source = df["llm_has_source"].mean()
    llm_has_source_std = df["llm_has_source"].std()
    llm_has_source_ci = 1.96 * llm_has_source_std / (len(df) ** 0.5)
    rag_has_source = df["rag_has_source"].mean()
    rag_has_source_std = df["rag_has_source"].std()
    rag_has_source_ci = 1.96 * rag_has_source_std / (len(df) ** 0.5)
    from datetime import timedelta
    return {
        "llm_acc": llm_acc,
        "llm_acc_std": llm_acc_std,
        "llm_acc_ci": llm_acc_ci,
        "rag_acc": rag_acc,
        "rag_acc_std": rag_acc_std,
        "rag_acc_ci": rag_acc_ci,
        "llm_time": llm_time,
        "llm_time_sum": str(timedelta(seconds=llm_time_sum)),
        "llm_time_std": llm_time_std,
        "rag_time": rag_time,
        "rag_time_sum": str(timedelta(seconds=rag_time_sum)),
        "rag_time_std": rag_time_std,
        "llm_has_source": llm_has_source,
        "llm_has_source_std": llm_has_source_std,
        "llm_has_source_ci": llm_has_source_ci,
        "rag_has_source": rag_has_source,
        "rag_has_source_std": rag_has_source_std,
        "rag_has_source_ci": rag_has_source_ci,
    }


calc_metrics(one_hop_true_false_v2_df)


# %%
