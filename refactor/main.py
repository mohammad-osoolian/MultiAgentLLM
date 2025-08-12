import utils
from time import sleep
import openai
from openai import OpenAI
from agent import *
import pandas as pd
from dataset import DataHandler
from logger import AgentLogger
from debate import Debate
from metrics import *
from config import *
from experiment import *


# classes = [
# 'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 
# 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
# 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 
# 'remorse', 'sadness', 'surprise', 'neutral',
# ]

# classes = [
#     'sadness', 'anger', 'love', 'surprise', 'joy', 'fear'
# ]

# data = DataHandler('../datasets/emotion.tsv', classes)
r1 = ResultHandler('../selected-experiments/debate-potential/goemotions/cognitive-persona-1/result/classification-result.tsv')
r2 = ResultHandler('../selected-experiments/debate-potential/goemotions/temper-2-with-description-3/result/classification-result.tsv')
r3 = ResultHandler('../selected-experiments/debate-potential/goemotions/temper-15/result/classification-result.tsv')
# r1 = ResultHandler('../experiments/number_pattern/temp15/result/classification-result.tsv')
# df4 = ResultHandler('../experiments/goemotions/exp4/results/zeroshot-result.tsv').df
# dfs = [df1, df2]

print(analysis_debate_potential([r1.predicts, r2.predicts, r2.predicts], r1.labels))

# predicts = pd.concat([df['predict'] for df in dfs] + [df['explanation'] for df in dfs], axis=1)
# predicts.columns = [f'predict_{i}' for i in range(len(dfs))] + [f'explanation_{i}' for i in range(len(dfs))]

# condition = ~predicts.eq(predicts.iloc[:, 0], axis=0).all(axis=1)
# id_label = dfs[0][['id', 'label']]
# diff_df = pd.concat([id_label, predicts], axis=1)[condition]

# client = OpenAI(api_key=utils.API_KEY)

# persona = (
#     "You are an expert in emotional analysis and language understanding. You deeply understand nuances in language and context, and you consider tone, implication, and word choice to identify the most appropriate emotional label"
# )

# instruction = (
#     "Your task is to classify the emotional content of a sentence into one of 28 emotion categories: "
#     "admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral\n"
#     "return your response as a JSON object with the following format:"
# )
# output_schema = (
#     "{\n"
#     '   "category": "<predicted category>"\n'
#     "}"
# )

# system_prompt = f"{persona}\n{instruction}\n{output_schema}"

# text = (
#     "Two responses from two other agents are provided for you.\n"
#     "use other agents responses as a hint and select the best answer"
#     'sentence: “Wow. Just .. wow. Read a book, sweatie!”'
#     "\agent1: disapproval\n"
#     "agent2: amusement\n"
#     "agent3: annoyance"
# )


# messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': text}]

# agent = LlmAgent('gpt-4o-mini', client, system_prompt=system_prompt, temperature=1)

# responses = agent.inference(messages, 10, response_format={ "type": "json_object" })
# print(*[choice.message.content for choice in responses.choices])