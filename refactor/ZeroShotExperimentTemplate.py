from dataset import *
from config import *
from experiment import *
from agent import *
import utils
import os
from openai import OpenAI
client = OpenAI(api_key=utils.API_KEY)
data = DataHandler(os.path.join('..', 'datasets', 'multi-label-emotion.tsv'))

def validator(predict):
    for l in predict['label_ids'].split(','):
        if not (0 < int(l) < 29):
            return False
    return 'label_ids' in predict and 'explanation' in predict

def extractor(predict):
    result = {}
    label_ids = list(map(int, predict['label_ids'].split(',')))
    result['predict'] = label_ids
    result['explanation'] = predict['explanation']
    return result   

config = ZeroShotConfig(utils.PERSONA, utils.INSTRUCTION, utils.DEBATE_OUTPUT_SCHEMA, validator, extractor)
agent = ZeroShotLlm('gpt-4o-mini', client, config)
basepath = os.path.join('..','experiments','multi-emotion','init-test')
experiment = ZeroShotExperiment(data, agent, basepath)

experiment.run()