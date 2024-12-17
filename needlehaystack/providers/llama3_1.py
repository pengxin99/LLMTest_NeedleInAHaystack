import os
from operator import itemgetter
from typing import Optional

# from openai import AsyncOpenAI
# from langchain_openai import ChatOpenAI  
# from langchain.prompts import PromptTemplate
# import tiktoken

import torch

from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from intel_extension_for_transformers.transformers import RtnConfig
from intel_extension_for_transformers.transformers.llm.utils.generation import _beam_search, _greedy_search
import argparse

from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
)


from .model import ModelProvider
model_dict ={
    "llama3.1-8B-Instruct-GPTQ": "C:\\Users\\Local_Admin\\Desktop\\workspace\\IPEX\\Models\\Meta-Llama-3.1-8B-Instruct-GPTQ",
}


device = "xpu"
torch_dtype = torch.float16


class Llama3_1(ModelProvider):
    """
    A wrapper class for interacting with OpenAI's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the OpenAI model to use for evaluations and interactions.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 300,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "llama3.1-8B-Instruct-GPTQ",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        Initializes the OpenAI model provider with a specific model.

        Args:
            model_name (str): The name of the OpenAI model to use. Defaults to 'gpt-3.5-turbo-0125'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """
        # api_key = os.getenv('NIAH_MODEL_API_KEY')
        # if (not api_key):
        #     raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = None
        model_path = model_dict[model_name]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            quantization_config=None,
            trust_remote_code=True,
            use_llm_runtime=False
        )
        model = model.to(torch.float16)
        self.model = model

    async def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        text = self.tokenizer.apply_chat_template(prompt,
                                                  tokenize=False,
                                                  add_generation_prompt=True,
                                                  **self.model_kwargs
                                                  )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        generate_kwargs = dict(do_sample=False, cache_implementation='static') #static cache to improve perf        
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=128, **generate_kwargs)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        response =  self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("-------------------- AI: ", response)
        
        return response
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the OpenAI model, and returns the model's response. This method leverages the LangChain 
        library to build a sequence of operations: extracting input variables, generating a prompt, 
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question. 
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a 
            dynamically provided question. The runnable encapsulates the entire process from prompt 
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """

        pass
    

