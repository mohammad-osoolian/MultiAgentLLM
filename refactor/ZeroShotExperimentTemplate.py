from dataset import *
from config import *
from experiment import *
from agent import *
import utils
from openai import OpenAI
client = OpenAI(api_key=utils.API_KEY)
data = DataHandler('../datasets/number_pattern copy.tsv')

def validator(predict):
    return 'next_number' in predict and 'explanation' in predict and predict['next_number'].isdigit()

def extractor(predict):
    result = {'predict':None, 'F2':None}
    result['predict'] = predict['next_number']
    result['F2'] = predict['explanation']
    return result

config = ZeroShotConfig(utils.PERSONA, utils.INSTRUCTION, utils.DEBATE_OUTPUT_SCHEMA, validator, extractor)
agent = ZeroShotLlm('gpt-4o-mini', client, config)
basepath = '../experiments/number_pattern/dev3'
experiment = ZeroShotExperiment(data, agent, basepath)

experiment.run()