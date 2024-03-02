from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_llm_chain():
    model_name_or_path = "TheBloke/Starling-LM-7B-alpha-AWQ"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        device_map="cuda:0"
    )

    generation_params = {
        "do_sample": False,
        "max_new_tokens": 4096,
        "repetition_penalty": 1.1
    }

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_params
    )


    hf = HuggingFacePipeline(pipeline=pipe)

    return hf
