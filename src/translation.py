import chainlit as cl


@cl.step
async def detect_language(content: str):
    return "en"


@cl.step
async def translation(content: str, target_language: str):
    return {
        "source_language": "en",
        "target_language": target_language,
        "translation": content
    }
