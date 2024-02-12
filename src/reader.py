from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class AbstractsReader(BaseReader):
    def __init__(
        self,
        *args: Any,
        pd_read_csv_kwargs: Optional[Dict] = None,
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._pd_read_csv_kwargs = pd_read_csv_kwargs or {}

    def load_data(
        self, file: Path, show_progress: bool = False
    ) -> List[Document]:
        """Parse file."""
        import pandas as pd
        df = pd.read_csv(file, **self._pd_read_csv_kwargs)
        df = df.dropna(subset=["abstract"])
        iterator = df.iterrows()
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(df))
        return [
            Document(text=row["abstract"], metadata={
                "PMID": str(row["PMID"]),
            }) for _, row in iterator
        ]


class FullArticleReader(BaseReader):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)

    def load_data(
        self, file: Path,
    ) -> List[Document]:
        """Parse file."""
        from bs4 import BeautifulSoup
        with open(file, "r") as f:
            text = f.read()
        soup = BeautifulSoup(text, "lxml")
        title = soup.find("article-title").get_text()
        abstract = soup.find("abstract").get_text()
        body = soup.find("body").get_text()

        text = title + "\n" + abstract + "\n" + body

        return [
            Document(text=text, metadata={
                "PMC": Path(file).stem.replace("PMC", "")
            })
        ]
