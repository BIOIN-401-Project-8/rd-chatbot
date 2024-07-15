from functools import cache  # function cache, speeds up recursive calls

import chainlit as cl
# https://deep-translator.readthedocs.io/en/latest/deep_translator.html
from deep_translator.base import BaseTranslator
from deep_translator import GoogleTranslator


@cache
def get_translator(translator:str="google"):
    '''
    return deep_translator BaseTranslator object 
    '''
    if translator == "google":
        return GoogleTranslator(source="auto", target="en")
    

def _translate(translator:BaseTranslator, content:str, source:str="auto", target:str="en"):
    '''
    translate content from source language to target language using 
    BaseTranslator object
    '''
    translator.source = source
    translator.target = target
    translated = translator.translate(content)
    return translated

@cl.step
async def translate(translator:BaseTranslator, content:str, source:str="auto", target:str="en"):
    return _translate(translator, content, source, target)