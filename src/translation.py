from functools import cache

import chainlit as cl
import hanzidentifier
from deep_translator import GoogleTranslator
from lingua import LanguageDetectorBuilder


@cache
def get_language_detector():
    return LanguageDetectorBuilder.from_all_spoken_languages().build()


@cache
def get_translator():
    return GoogleTranslator(source="auto", target="en")


def _detect_language(content: str):
    detector = get_language_detector()
    detected_langauge = detector.detect_language_of(content)
    iso_code = detected_langauge.iso_code_639_1.name if detected_langauge else None
    iso_code = iso_code.lower() if iso_code else None
    if iso_code == "zh":
        if hanzidentifier.is_traditional(content):
            iso_code = "zh-TW"
        else:
            iso_code = "zh-CN"
    return iso_code


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
