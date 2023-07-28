import argparse
import time
import torch
import yaml
from langchain import PromptTemplate
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizer, pipeline)

"""
Ad-hoc sanity check to see if model outputs something coherent
Not a robust inference platform!
"""

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")



def get_llm_response(prompt):
    # time the generation
    start_time = time.time()
    print(f"About to generate response to prompt: {prompt}")
    raw_output = pipe(prompt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_ms = int(elapsed_time * 1000)
    print(f"Generation complete in {total_ms} ms: "+raw_output[0]["generated_text"].replace(prompt,""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()

    config = read_yaml_file(args.config_path)

    print("Load model")
    device = torch.device('cpu')
    model_path = f"{config['model_output_dir']}/{config['model_name']}"
    if "model_family" in config and config["model_family"] == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path, device_map="cpu")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)
    if not tokenizer.pad_token:
        # Add padding token if missing, e.g. for llama tokenizer
        #tokenizer.pad_token = tokenizer.eos_token  # https://github.com/huggingface/transformers/issues/22794
        num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'}) #, 'additional_special_tokens': ['user: ','assistant: ', 'system: ']}, replace_additional_special_tokens=False)
        print(f"added {num_added_toks} extra tokens")
        if num_added_toks > 0:
            model.resize_token_embeddings(len(tokenizer))

    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=512,
        repetition_penalty=1.15,
        device=-1,
        penalty_alpha=0.5,
        top_k=8,
        return_full_text=False
    )

    get_llm_response("user: Hi, how are you?\nassistant: ")
    get_llm_response("user: Tell me a touching story about a happy couple who live in San Francisco and run a theater together.\nassistant: ")