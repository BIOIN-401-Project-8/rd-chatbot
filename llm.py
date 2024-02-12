from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_llm_chain():
    model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=False,
        revision="gptq-4bit-32g-actorder_True"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=8192,
        do_sample=False,
        repetition_penalty=1.1
    )

    hf = HuggingFacePipeline(pipeline=pipe)

    return hf
