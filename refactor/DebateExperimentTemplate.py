from dataset import *
from config import *
from experiment import *
from agent import *
from debate import Debate
import utils
from openai import OpenAI

classes = [
'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 
'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 
'remorse', 'sadness', 'surprise', 'neutral',
]

def validator(predict):
    return 'category' in predict and 'explanation' in predict and predict['category'] in classes

def extractor(predict):
    result = {'predict': '', 'explanation': ''}
    if predict is not None: 
        result['predict'] = predict['category']
        result['explanation'] = predict['explanation']
    return result 

persona = (
    "You are an expert in emotional analysis and language understanding. You deeply understand nuances in language and context, and you consider tone, implication, and word choice to identify the most appropriate emotional label"
)

instruction = (
    "Your task is to classify the emotional content of a sentence into one of 28 emotion categories and explain why. here are the list of categories: "
    "admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral\n"
    "return your response as a JSON object with the following format:"
)
output_schema1 = (
    "{\n"
    '   "category": "<predicted category>",\n'
    '   "explanation": "<your explanation>"\n'
    "}"
)

output_schema2 = (
    "{\n"
    '   "category": "<predicted category>",\n'
    '   "explanation": "<your explanation>"\n'
    "}"
)

data = DataHandler(os.path.join('../datasets', 'debate_goemotions_single_labeled.tsv'))
client = OpenAI(api_key=utils.API_KEY)
config1 = DebateAgentConfig(persona, instruction, output_schema1, validator, extractor)
config2 = DebateAgentConfig(persona, instruction, output_schema2, validator, extractor)

agent1 = DebateAgent('gpt-4o-mini', client, config1, temperature=1)
agent2 = DebateAgent('gpt-4o-mini', client, config2, temperature=1)
# agent3 = DebateAgent('gpt-4o-mini', client, config, temperature=1)

# debate = DebateExperiment(data, [agent1, agent2], '../experiments/debate/number_pattern/same-agents-4o-mini')
debate = DebateExperiment(data, [agent1, agent2], os.path.join('../experiments', 'debate', 'goemotions', 'repeat-answer'))
debate.run()