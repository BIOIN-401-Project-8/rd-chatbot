from functools import lru_cache

import hanziconv
import torch
from deep_translator.base import BaseTranslator
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

SEAMLESS_M4T_V2_LANGUAGES_TO_CODES = {
    "english": "en",
    "french": "fr",
    "italian": "it",
    "chinese (simplified)": "zh-CN",
    "chinese (traditional)": "zh-TW",
    # TODO: add remaining languages
    # https://huggingface.co/facebook/seamless-m4t-v2-large
}

SEAMLESS_M45_V2_MAPPING = {
    "en": "eng",
    "fr": "fra",
    "it": "ita",
    "zh-CN": "cmn",
    "zh-TW": "cmn_Hant",
}


class SeamlessM4Tv2Translator(BaseTranslator):
    def __init__(self, source: str = "fr", target: str = "en", **kwargs):
        self.model_name = None
        super().__init__(languages=SEAMLESS_M4T_V2_LANGUAGES_TO_CODES, source=source, target=target, **kwargs)

    @lru_cache(maxsize=2)
    def get_model_processor(self):
        model_name = "facebook/seamless-m4t-v2-large"
        processor = AutoProcessor.from_pretrained(model_name)
        model = SeamlessM4Tv2ForTextToText.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(self.device)
        return model, processor

    def translate(self, text: str, **kwargs) -> str:
        model, processor = self.get_model_processor()
        inputs = processor([text], src_lang=SEAMLESS_M45_V2_MAPPING[self.source], return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        output_tokens = model.generate(**inputs, tgt_lang=SEAMLESS_M45_V2_MAPPING[self.target])
        res = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)

        return res
