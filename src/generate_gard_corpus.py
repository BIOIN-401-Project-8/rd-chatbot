from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from gard import GARD
from tqdm.contrib.concurrent import thread_map

gard = GARD()


def get_text(gard_id):
    name = gard[gard_id]["name"]
    url = gard.get_url(gard_id)
    summary = "This section is currently not found."
    try:
        response = requests.get(url).text
        soup = BeautifulSoup(response, "html.parser")
        summary = soup.select_one("#disease > div > div > div.row.py-5 > div.col-12.col-xl-6.left-side > app-disease-at-a-glance-summary > div > div.mobile.d-lg-none.mb-5 > clamper > div > span.host.fs-md-18").text
    except Exception:
        pass
    text = f"Summary of {name}: {summary}"
    return gard_id, text


def main():
    items = thread_map(get_text, list(gard.map), max_workers=64)
    texts = [item[1] for item in sorted(items, key=lambda x: x[0])]

    df = pd.DataFrame(texts, columns=["text"])
    Path("eval/data/RD").mkdir(parents=True, exist_ok=True)
    df.to_csv("eval/data/RD/gard_corpus.csv", index=False)


if __name__ == "__main__":
    main()
