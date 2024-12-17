import os
from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
# from langchain_community.chat_models import ChatOpenAI
# from ..config.private_config import OPENAI_API_KEY
from ..config.criteria import CRITERIA_NEEDLEHAYSTACK, CRITERIA_EXAM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import AutoModel, pipeline
import torch
import json
from ..utils import LanguageDetector

device_map = 'xpu'
device = "xpu"

model_dict ={
    "llama3.1-8B-Instruct-GPTQ": "C:\\Users\\Local_Admin\\Desktop\\workspace\\IPEX\\Models\\Meta-Llama-3.1-8B-Instruct-GPTQ",
    "qwen2.5-3B": "C:\\Users\\Local_Admin\\Desktop\\workspace\\IPEX\\Qwen2.5\\Qwen2.5-3B-Instruct"
}

class Llama3_1_Evaluator(Evaluator):
    
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 1024,
                                      temperature = 0)
    
    def __init__(self,
                 model_name: str = "qwen2.5-3B",
                 true_answer: str = '',
                 question_asked: str = '',
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked
        
        self.api_key = None
        self.model_path = model_dict[model_name]
        
        
        # self.evaluator = AutoModelForCausalLM.from_pretrained(model_path,
        #                                                   device_map=device_map,
        #                                                   torch_dtype=torch.float16)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.CRITERIA = CRITERIA_NEEDLEHAYSTACK
    
    
    def generate_answer(self,  response: str, true_answer: str, question_asked: str, criteria: str) -> list[dict[str, str]]:
        print("response:", response)
        print("true_answer:", true_answer)
        print("question_asked:", question_asked)
        print("criteria:", criteria)
        
        return [
        {"role": "system",
        #  "content": """You are an evaluator AI designed to assess how closely the response matches the Correct Answer, the response do not need to add any value or detail beyond the Correct answer. You will be given a correct answer, a model's response, and specific criteria for evaluation. Your task is to provide a score based on how well the model's response meets the criteria, Scores are 10 when the response is the same as the Correct answer. Keep your assessment concise and objective."""
         "content": """You are an evaluator AI designed to assess how closely the response matches the Correct Answer, You will be given a correct answer, a model's response, and specific criteria for evaluation. Your task is to provide a score based on how well the model's response meets the criteria, Scores are 10 when the response is the same as the Correct answer. Please always give score 10 if the model response is the same or very closely as the correct answer."""
        },
        {"role": "user",
         "content": f"""
        Question:
        --- --- ---
        {question_asked}
        Correct Answer:
        --- --- ---
        {true_answer}

        Model Response:
        --- --- ---
        {response}
        Evaluation Criteria:
        --- --- ---
        {criteria}

        Based on the content above,  briefly explain the reason for your score. 
        result must in dict form and contain reasoning and score.
        final output must use the markdown format, as below:
        ```json
        {{
            "score": 1-10,
            "reasoning": "..."
        
        }}
        ```
        """
                },
     
        ]
    
    async def evaluate_response_async(self,  response: str, true_answer: str, question_asked: str) -> dict:
        prompt = self.generate_answer(response , true_answer, question_asked, self.CRITERIA)
        text = self.tokenizer.apply_chat_template(prompt,
                                                  tokenize=False,
                                                  add_generation_prompt=True,
                                                  **self.model_kwargs
                                                  )
        self.evaluator = AutoModelForCausalLM.from_pretrained(self.model_path,
                                            device_map=device_map,
                                            torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        generate_kwargs = dict(do_sample=False, cache_implementation='static') #static cache to improve perf        
        generated_ids = self.evaluator.generate(model_inputs.input_ids, max_new_tokens=32, **generate_kwargs)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        eval_response =  self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        self.evaluator = None
        self.tokenizer = None
        return eval_response
        # try:
        #     eval_result = json.loads(eval_response)
        #     # print('*'*50)
        #     # print(eval_result)
        #     return eval_result
        # except json.JSONDecodeError as e:
        #     print('-'*50)
        #     print(f"response: {response}")
        #     print(f"true_answer: {true_answer}")
        #     print(f"question_asked: {question_asked}")
        #     print('='*50)
        #     print(f"Error decoding json: {e}")
        #     print(f"eval_response: {eval_response}")
        #     return {'reasoning': 'Error decoding json', 'score': 0}
    
    
    def evaluate_response(self, response: str) -> int:
        prompt = self.generate_answer(response , self.true_answer, self.question_asked, self.CRITERIA)

        print("------ load eval model")
        self.evaluator = AutoModelForCausalLM.from_pretrained(self.model_path,
                                            device_map=device_map,
                                            torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        print("------ evaluate_response prompt: ", prompt)
        text = self.tokenizer.apply_chat_template(prompt,
                                                  tokenize=False,
                                                  add_generation_prompt=True,
                                                  **self.model_kwargs
                                                  )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        # generate_kwargs = dict(do_sample=False, cache_implementation='static') #static cache to improve perf        
        generate_kwargs = dict(do_sample=False)      
        generated_ids = self.evaluator.generate(model_inputs.input_ids, max_new_tokens=256, **generate_kwargs)

        # generated_ids = self.evaluator.generate(model_inputs.input_ids,
        #                                max_new_tokens=1024)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        eval_response =  self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.evaluator = None
        self.tokenizer = None

        eval_result = LanguageDetector.markdown_to_json(eval_response)
        print('eval_result:', eval_result)
        print('type(eval_result):', type(eval_result))
        if isinstance(eval_result, str):
            eval_result = json.loads(eval_result)
            return int(eval_result['score']), eval_result['reasoning']
        elif isinstance(eval_result, dict):
            return int(eval_result['score']), eval_result['reasoning']
        
        else:
            return 0, 'Error decoding json'
    

    # def evaluate_response(self, response: str) -> int:
    #     evaluator = load_evaluator(
    #         "labeled_score_string",
    #         criteria=self.CRITERIA,
    #         llm=self.evaluator,
    #     )

    #     eval_result = evaluator.evaluate_strings(
    #         # The models response
    #         prediction=response,

    #         # The actual answer
    #         reference=self.true_answer,

    #         # The question asked
    #         input=self.question_asked,
    #     )

    #     return int(eval_result['score'])