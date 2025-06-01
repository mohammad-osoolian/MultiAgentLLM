from concurrent.futures import ThreadPoolExecutor
import json
from config import *
from tqdm import tqdm
import utils
from utils import parse_json_or_none
from ratelimit import limits, sleep_and_retry
from logger import *


class LlmAgent:
    def __init__(self, model, client, logger:AgentLogger=NullLogger(), system_prompt=utils.DEFAULT_SYSTEM_PROMPT, temperature=1, rpm_limit=5):
        self.model = model
        self.logger = logger
        self.temperature = temperature
        self.client = client
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logger.log_messages([self.messages[0]])
    
    def inference(self, messages, n=1, response_format={ "type": "text" }):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            n=n,
            response_format=response_format,
            max_tokens=utils.DEFAULT_MAX_TOKENS,
        )
        return response
    
    def batch_inference(self, batch, rpm_limit=utils.RPM_LIMIT, n=1, response_format={"type": "text"}):
        @sleep_and_retry
        @limits(calls=rpm_limit, period=60)
        def rate_limited_inference(messages):
            return self.inference(messages, n=n, response_format=response_format)
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(rate_limited_inference, batch))
        return results
    
    def message(self, text, keep_history=True, response_format={"type": "text"}):
        messages = self.messages if keep_history else self.messages.copy()
        messages.append({"role": "user", "content": text})
        response = self.inference(messages, response_format=response_format).choices[0].message
        role = response.role
        content = response.content
        messages.append({"role": role, "content": content})
        self.logger.log_messages(messages[-2:])
        return content

    def info(self):
        info = dict()
        info['model'] = self.model
        info['temperature'] = self.temperature
        info['system_prompt'] = self.system_prompt
        return info
      

class ZeroShotLlm(LlmAgent):
    def __init__(self, model, client, config: ZeroShotConfig, logger=NullLogger(), temperature=1):
        self.config = config
        system_prompt = config.build_system_prompt()
        self.validate_prediction = config.validator
        self.extract_prediction = config.extractor
        super().__init__(model, client, logger, system_prompt=system_prompt, temperature=temperature)
    
    def predict(self, text, keep_history=False):
        """
        returns predicted answer and Error
        """
        response = self.message(text, keep_history, response_format={'type': 'json_object'})
        prediction, err = self.convert_prediction(response)

        return prediction, err
    
    def batch_predict(self, batch,rpm_limit=utils.RPM_LIMIT):
        @sleep_and_retry
        @limits(calls=rpm_limit, period=60)
        def rate_limited_predict(text):
            return self.predict(text)
        
        with ThreadPoolExecutor() as executor:
            results = results = list(tqdm(executor.map(rate_limited_predict, batch), total=len(batch)))
        return results
        
    
    def validate_prediction(self, pred):
        pass

    def extract_prediction(self, pred):
        pass

    def convert_prediction(self, json_predict):
        dict_predict, err = parse_json_or_none(json_predict)
        if err or (not self.validate_prediction(dict_predict)): 
            return None, 1
        return self.extract_prediction(dict_predict), 0

    
class DebateAgent(ZeroShotLlm):
    def __init__(self, model, client, config, logger=NullLogger(), temperature=1):
        super().__init__(model, client, config, logger, temperature)

    def clean(self):
        self.messages = [self.messages[0]]
        
    def update_answer(self, predicts, expls, errors, text, agent_index):
        update_answer_prompt = self.build_update_answer_prompt(predicts, expls, errors, text, agent_index)
        response = self.message(update_answer_prompt, response_format={'type': 'json_object'})
        result, err = self.convert_prediction(response)
        return result, err

    def build_update_answer_prompt(self, predicts, expls, errors, text, agent_index):
        your_answer = (
            "Your Previous Answer\n"
            f"Predictoin: {predicts[agent_index]}\n"
            f"Explanation: {expls[agent_index]}\n"
            )
        allanswers = ["Other Agent Answers"]
        for i in range(len(predicts)):
            if i == agent_index:
                continue
            answer = (
                f"Agent {i+1}\n"
                f"Predictoin: {predicts[i]}\n"
                f"Explanation: {expls[i]}\n"
            )
            allanswers.append(answer)
        prompt = text + '\n' + utils.UPDATE_ANSWER_MESSAGE + your_answer + '\n'.join(allanswers)
        return prompt