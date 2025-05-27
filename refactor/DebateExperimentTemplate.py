from dataset import *
from config import *
from experiment import *
from agent import *
from debate import Debate
import utils
from openai import OpenAI


def validator(predict):
    return 'next_number' in predict and 'explanation' in predict and predict['next_number'].isdigit()

def extractor(predict):
    result = {'predict':None, 'explanation':None}
    result['predict'] = predict['next_number']
    result['explanation'] = predict['explanation']
    return result

data = DataHandler('../datasets/number_pattern.tsv')
client = OpenAI(api_key=utils.API_KEY)
config = DebateAgentConfig(utils.PERSONA, utils.INSTRUCTION, utils.DEBATE_OUTPUT_SCHEMA, validator, extractor)
agent = DebateAgent('gpt-4o-mini', client, config)
# basepath = '../experiments/debate/number_pattern/test-1'

config = DebateAgentConfig(utils.PERSONA, utils.INSTRUCTION, utils.DEBATE_OUTPUT_SCHEMA, validator, extractor)
agent1 = DebateAgent('gpt-4o-mini', client, config, temperature=1)
agent2 = DebateAgent('gpt-4o-mini', client, config, temperature=1)
agent3 = DebateAgent('gpt-4o-mini', client, config, temperature=1)

debate = DebateExperiment(data, [agent1, agent2, agent3], '../experiments/debate/number_pattern/same-agents-4o-mini')
debate.run()