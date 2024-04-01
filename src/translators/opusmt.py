from functools import lru_cache

import hanziconv
import torch
from deep_translator.base import BaseTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

OPUS_MT_LANGUAGES_TO_CODES = {
    "afrikaans": "af",
    "albanian": "sq",
    "arabic": "ar",
    "armenian": "hy",
    "azerbaijani": "az",
    "basque": "eu",
    "bislama": "bi",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese (simplified)": "zh-CN",
    "chinese (traditional)": "zh-TW",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "ewe": "ee",
    "fijian": "fj",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "ganda": "lg",
    "german": "de",
    "haitian": "ht",
    "hausa": "ha",
    "hindi": "hi",
    "hiri motu": "ho",
    "hungarian": "hu",
    "icelandic": "is",
    "igbo": "ig",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "kinyarwanda": "rw",
    "kongo": "kg",
    "kuanyama": "kj",
    "lingala": "ln",
    "luba-katanga": "lu",
    "macedonian": "mk",
    "malagasy": "mg",
    "malayalam": "ml",
    "maltese": "mt",
    "manx": "gv",
    "marathi": "mr",
    "marshallese": "mh",
    "ndonga": "ng",
    "nyanja": "ny",
    "oromo": "om",
    "rundi": "rn",
    "russian": "ru",
    "samoan": "sm",
    "sango": "sg",
    "shona": "sn",
    "slovak": "sk",
    "southern sotho": "st",
    "spanish": "es",
    "swati": "ss",
    "swedish": "sv",
    "tagalog": "tl",
    "tigrinya": "ti",
    "tonga": "to",
    "tsonga": "ts",
    "tswana": "tn",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
    "welsh": "cy",
    "xhosa": "xh",
}


class OpusMTTranslator(BaseTranslator):
    def __init__(self, source: str = "fr", target: str = "en", **kwargs):
        self.model_name = None
        super().__init__(languages=OPUS_MT_LANGUAGES_TO_CODES, source=source, target=target, **kwargs)

    @lru_cache(maxsize=2)
    def get_model_tokenizer(self, source: str = "fr", target: str = "en"):
        if target.startswith("zh"):
            target = "zh"
        model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(self.device)
        return model, tokenizer

    def translate(self, text: str, **kwargs) -> str:
        model, tokenizer = self.get_model_tokenizer(self.source, self.target)
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        translated = model.generate(**inputs)
        res = tokenizer.decode(translated[0], skip_special_tokens=True)

        if self.target == "zh-TW":
            res = hanziconv.HanziConv.toTraditional(res)
        elif self.target == "zh-CN":
            res = hanziconv.HanziConv.toSimplified(res)

        return res
