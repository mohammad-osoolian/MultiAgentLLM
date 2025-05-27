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
client = OpenAI(api_key=utils.API_KEY)
# agent = LlmClassifier('gpt-4o-mini', client, classes, temperature=2)

# experiment = Classification(data, agent, '../experiments/emotion/temper-2-3/')
# experiment.run()

# r1 = ResultHandler('../experiments/number_pattern/temp1/result/classification-result.tsv')
# r2 = ResultHandler('../experiments/number_pattern/temp15/result/classification-result.tsv')
# r3 = ResultHandler('../experiments/number_pattern/gpt35/result/classification-result.tsv')
# labels = r1.labels

# analysis_debate_potential(r1.predicts, r2.predicts, r3.predicts, labels)

def validator(predict):
    return 'next_number' in predict and 'explanation' in predict and predict['next_number'].isdigit()

# def extractor(predict):
#     return predict['next_number']

# def extractor(predict):
#     result = {'predict':None, 'explanation':None}
#     result['predict'] = predict['next_number']
#     result['explanation'] = predict['explanation']
#     return result

# data = DataHandler('../datasets/number_pattern copy.tsv')
# client = OpenAI(api_key=utils.API_KEY)
# config = DebateAgentConfig(utils.PERSONA, utils.INSTRUCTION, utils.DEBATE_OUTPUT_SCHEMA, validator, extractor)
# agent1 = DebateAgent('gpt-4o-mini', client, config, temperature=1)
# agent2 = DebateAgent('gpt-4o-mini', client, config, temperature=1)
# agent3 = DebateAgent('gpt-4o-mini', client, config, temperature=1)

# debate = Debate(data, [agent1, agent2, agent3])
# debate.run()