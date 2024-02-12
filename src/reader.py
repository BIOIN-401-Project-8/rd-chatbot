from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class AbstractCSVReader(BaseReader):
    def __init__(
        self, *args: Any, extra_info: Optional[Dict] = None, pd_read_csv_kwargs: Optional[Dict] = None, **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._pd_read_csv_kwargs = pd_read_csv_kwargs or {}

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None, show_progress: bool = False
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
                "extra_info": extra_info or {}
            }) for _, row in iterator
        ]


class FullArticleXMLReader(BaseReader):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        from bs4 import BeautifulSoup
        with open(file, "r") as f:
            text = f.read()
        soup = BeautifulSoup(text, "html.parser")
        title = soup.find("article-title")
        title_text = title.get_text() if title else ""
        abstract = soup.find("abstract")
        abstract_text = abstract.get_text() if abstract else ""
        body = soup.find("body")
        body_text = body.get_text() if body else ""

        text = f"{title_text}\n{abstract_text}\n{body_text}"

        return [
            Document(text=text, metadata={
                "PMC": Path(file).stem.replace("PMC", ""),
                "extra_info": extra_info or {}
            })
        ]
