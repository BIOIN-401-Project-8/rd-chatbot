from deep_translator.base import BaseTranslator
from llama_index.core.base.llms.base import BaseLLM


class LLMTranslator(BaseTranslator):
    def __init__(self, llm: BaseLLM, languages: dict, source: str = "auto", target: str = "en", **kwargs):
        """
        @param source: source language to translate from
        @param target: target language to translate to
        """
        super().__init__(languages=languages, source=source, target=target, **kwargs)
        self._llm = llm

    def translate(self, text: str, **kwargs) -> str:
        return self.llm.complete(text).text
