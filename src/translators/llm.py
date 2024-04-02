from deep_translator.base import BaseTranslator
from llama_index.core.base.llms.base import BaseLLM


class LLMTranslator(BaseTranslator):
    def __init__(self, llm: BaseLLM, languages: dict, source: str = "auto", target: str = "en", **kwargs):
        """
        @param source: source language to translate from
        @param target: target language to translate to
        """
        super().__init__(languages=languages, source=source, target=target, **kwargs)
        self.llm = llm
        self.reverse_languages = {v: k.title() for k, v in languages.items()}

    def translate(self, text: str, **kwargs) -> str:
        source = self.reverse_languages[self.source]
        target = self.reverse_languages[self.target]

        response = self.llm.complete(
            f"Translate the {source} text to {target}. \nText: {text}\nTranslation: "
        )
        text = response.text
        text = text.removeprefix(f"{target}: ")
        return text
