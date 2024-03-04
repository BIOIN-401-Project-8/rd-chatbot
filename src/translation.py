from functools import cache

import chainlit as cl
import hanzidentifier
from deep_translator import GoogleTranslator
from lingua import Language, LanguageDetectorBuilder


@cache
def get_language_detector():
    return LanguageDetectorBuilder.from_all_spoken_languages().build()


@cache
def get_translator():
    return GoogleTranslator(source="auto", target="en")


def _detect_language(content: str, threshold: float = 0.5):
    detector = get_language_detector()
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
    confidence_values_dict = {confidence_value.language.name: confidence_value.value for confidence_value in confidence_values[:5]}
    return {
        "language": iso_code,
        "confidence_values": confidence_values_dict,
    }


@cl.step
async def detect_language(content: str):
    return _detect_language(content)


def _translate(content: str, source: str = "auto", target: str = "en"):
    translator = get_translator()
    translator.source = source
    translator.target = target
    return translator.translate(content)


@cl.step
async def translate(content: str, source: str = "auto", target: str = "en"):
    return _translate(content, source, target)
