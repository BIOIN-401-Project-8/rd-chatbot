import unittest

from src.reader import AbstractsReader, FullArticleReader

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

class TestAbstractsReader(unittest.TestCase):
    def test_load_data(self):
        reader = AbstractsReader()
        documents = reader.load_data(file=DATA_DIR / "abstracts.csv")
        self.assertEquals(len(documents), 2)
        self.assertEquals(documents[0].text, "Scanning electron microscopy of unmanipulated erythrocytes from patients with myotonic dystrophy or Duchenne dystrophy and patients who were Duchenne carriers showed a large increase in the number of stomatocytes over the number in normal controls. No specific morhologic changes that would differentiate any of the dystrophic patients from one another were seen. Adverse conditions such as washing before fixation or extreme pH produced a greater change in erythrocytes from these patients than in those from normal controls.")
        self.assertEquals(documents[0].metadata["PMID"], "3154")


class TestFullArticleReader(unittest.TestCase):
    def test_load_data(self):
        reader = FullArticleReader()
        documents = reader.load_data(file="/data/pmc-open-access-subset/6291/PMC10620460.xml")
        document = documents[0]
        self.assertEqual(len(document.text), 54310)
        self.assertEqual(document.metadata["PMC"], "10620460")



if __name__ == "__main__":
    unittest.main()
