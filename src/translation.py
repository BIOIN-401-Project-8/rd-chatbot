import os
from functools import cache

import chainlit as cl
import hanzidentifier
from deep_translator import GoogleTranslator
from deep_translator.base import BaseTranslator
from lingua import IsoCode639_1, Language, LanguageDetector, LanguageDetectorBuilder
from llama_index.llms.groq import Groq

from translators.llm import LLMTranslator
from translators.opusmt import OpusMTTranslator
from translators.seamlessm4tv2 import SeamlessM4Tv2Translator


@cache
def get_language_detector(*iso_codes: IsoCode639_1):
    return LanguageDetectorBuilder.from_iso_codes_639_1(*iso_codes).build()


@cache
def get_translator(translator: str = "google"):
    if translator == "google":
        return GoogleTranslator(source="auto", target="en")
    elif translator == "opusmt":
        return OpusMTTranslator(source="fr", target="en")
    elif translator == "seamlessm4tv2":
        return SeamlessM4Tv2Translator(source="fr", target="en")
    elif translator == "mixtral8x7b":
        os.environ["OPENAI_API_KEY"] = "None"
        llm = Groq(
            model="mixtral-8x7b-32768",
            api_key=os.environ["GROQ_API_KEY"],
        )
        languages = {
            "english": "en",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "spanish": "es",
        }
        return LLMTranslator(llm, languages, source="fr", target="en")
    elif translator == "mixtral_8x7b-instruct-v0.1-q4_0":
        os.environ["OPENAI_API_KEY"] = "None"
        from llama_index.llms.ollama import Ollama
        llm = Ollama(
            model="mixtral:8x7b-instruct-v0.1-q4_0",
            base_url="http://ollama:11434",
            request_timeout=30.0,
            temperature=0.0,
        )
        languages = {
            "english": "en",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "spanish": "es",
        }
        return LLMTranslator(llm, languages, source="fr", target="en")

def _detect_language(detector: LanguageDetector, content: str, threshold: float = 0.5):
    confidence_values = detector.compute_language_confidence_values(content)
    language = Language.ENGLISH
    for confidence_value in confidence_values:
        if confidence_value.value > threshold:
            language = confidence_value.language
            break
    iso_code = language.iso_code_639_1.name if language else None
    iso_code = iso_code.lower() if iso_code else None
    if iso_code == "zh":
        if hanzidentifier.is_traditional(content):
            iso_code = "zh-TW"
        else:
            iso_code = "zh-CN"
    confidence_values_dict = {
        confidence_value.language.name: confidence_value.value for confidence_value in confidence_values[:5]
    }
    return {
        "language": iso_code,
        "confidence_values": confidence_values_dict,
    }


@cl.step
async def detect_language(detector: LanguageDetector, content: str):
    return _detect_language(detector, content)


def _translate(translator: BaseTranslator, content: str, source: str = "auto", target: str = "en"):
    translator.source = source
    translator.target = target
    translated = translator.translate(content)
    return translated


@cl.step
async def translate(translator: BaseTranslator, content: str, source: str = "auto", target: str = "en"):
    return _translate(translator, content, source, target)
