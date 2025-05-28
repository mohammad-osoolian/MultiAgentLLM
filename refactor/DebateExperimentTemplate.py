from dataset import *
from config import *
from experiment import *
from agent import *
from debate import Debate
import utils
from openai import OpenAI


def validator(predict):
    for l in predict['label_ids'].split(','):
        if not (0 < int(l) < 29):
            return False
    return 'label_ids' in predict and 'explanation' in predict

def extractor(predict):
    result = {}
    label_ids = sorted(list(map(int, predict['label_ids'].split(','))))
    result['predict'] = label_ids
    result['explanation'] = predict['explanation']
    return result   

data = DataHandler(os.path.join('datasets', 'multi-label-emotion.tsv'))
client = OpenAI(api_key=utils.API_KEY)
config = DebateAgentConfig(utils.PERSONA, utils.INSTRUCTION, utils.DEBATE_OUTPUT_SCHEMA, validator, extractor)
# basepath = '../experiments/debate/number_pattern/test-1'

config = DebateAgentConfig(utils.PERSONA, utils.INSTRUCTION, utils.DEBATE_OUTPUT_SCHEMA, validator, extractor)
agent1 = DebateAgent('gpt-4o-mini', client, config, temperature=1.2)
agent2 = DebateAgent('gpt-4o-mini', client, config, temperature=1)
# agent3 = DebateAgent('gpt-4o-mini', client, config, temperature=1)

# debate = DebateExperiment(data, [agent1, agent2], '../experiments/debate/number_pattern/same-agents-4o-mini')
debate = DebateExperiment(data, [agent1, agent2], os.path.join('experiments', 'debate', 'multi-emotion', 'init-test-9'))
debate.run()