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


classes = [
'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 
'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 
'remorse', 'sadness', 'surprise', 'neutral',
]


# classes = [
#     'sadness', 'anger', 'love', 'surprise', 'joy', 'fear'
# ]

# data = DataHandler('../datasets/emotion.tsv', classes)
client = OpenAI(api_key=utils.API_KEY)

persona = (
    "You are an expert in emotional analysis and language understanding. You deeply understand nuances in language and context, and you consider tone, implication, and word choice to identify the most appropriate emotional label"
)

instruction = (
    "Your task is to classify the emotional content of a sentence into one of 28 emotion categories and explain why. here are the list of categories: "
    "admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral\n"
    "return your response as a JSON object with the following format:"
)
output_schema = (
    "{\n"
    '   "category": "<predicted category>"\n'
    # '   "explanation": "<your explanation>"\n'
    "}"
)

def validator(predict):
    return 'category' in predict and predict['category'] in classes

def extractor(predict):
    result = {'predict': ''}
    if predict is not None: 
        result['predict'] = predict['category']
    return result 

config = ZeroShotConfig(persona, instruction, output_schema, validator, extractor)

agent = ZeroShotLlm('gpt-4o-mini', client, config, temperature=1.5)
data = DataHandler(os.path.join('..', 'datasets', 'goemotions_single_labeled.tsv'))
# experiment = ZeroShotExperiment(data, agent, '../experiments/goemotions/exp-100-4')
# experiment.run()

df1 = ResultHandler('../experiments/goemotions/exp-100-1/results/zeroshot-result.tsv').df
df2 = ResultHandler('../experiments/goemotions/exp-100-2/results/zeroshot-result.tsv').df
df3 = ResultHandler('../experiments/goemotions/exp-100-3/results/zeroshot-result.tsv').df
df4 = ResultHandler('../experiments/goemotions/exp-100-4/results/zeroshot-result.tsv').df
dfs = [df1, df2,df3,df4]

predicts = pd.concat([df['predict'] for df in dfs],  axis=1)
predicts.columns = [f'predict_{i}' for i in range(len(dfs))]

condition = ~predicts.eq(predicts.iloc[:, 0], axis=0).all(axis=1)
id_label = dfs[0][['id', 'label']]
diff_df = pd.concat([id_label, predicts], axis=1)[condition]
diff_df = diff_df[~(diff_df[predicts.columns].isna().sum(axis=1).isin([2, 3, 4]))]
print(diff_df, len(diff_df))

fulldata = DataHandler('/home/linser/Me/MultiAgentLLM/Datasets/goEmotions/goemotions_single_labeled.tsv').df
fd = fulldata[fulldata['id'].isin(diff_df['id'].tolist())]
print(fd)
fd.to_csv('../datasets/temp.tsv', sep='\t', index=False)