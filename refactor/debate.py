from agent import DebateAgent
from experiment import *
from dataset import *
class Debate:
    def __init__(self, data:DataHandler, agents: list[DebateAgent]):
        self.data = data
        self.agents = agents
        self.predictions = []
        self.final_predictions = []

    def check_agreement(self, predicts):
        return len(set(predicts)) == 1

    def final_predict(self, predicts):
        freq = {}
        for p in predicts:
            freq[p] = freq.get(p, 0) + 1
        return max(freq, key=freq.get)
    
    def run(self):
        self.single_agent()
        self.debate()
    
    def single_agent(self):
        single_agent_results = []
        for agent in self.agents:
            agent_results = agent.batch_predict(self.data.texts)
            single_agent_results.append(agent_results)
        for i in range(len(self.data.texts)):
            sample = {'predicts':[], 'explanations':[], 'errors':[]}
            for j in range(len(self.agents)):
                result, err = single_agent_results[j][i]
                pred, expl = (result['predict'], result['explanation']) if not err else ("", "")
                sample['errors'].append(err)
                sample['predicts'].append(pred)
                sample['explanations'].append(expl)
            self.predictions.append(sample)
    
    def debate(self):
        for sample in self.predictions:
            predicts, expls, errors = sample['predicts'], sample['explanations'], sample['errors']
            print(predicts)
            if self.check_agreement(predicts):
                self.final_predictions.append(self.final_predict(predicts))
                print(self.final_predictions[-1])
                continue
            print("NEED DEBATE")
            for i in range(3):
                predicts, expls, errors = self.debate_round(predicts, expls, errors)
                print(predicts)
                if self.check_agreement(predicts):
                    self.final_predictions.append(self.final_predict(predicts))
                    print(self.final_predictions[-1])
                    break
    
    def debate_round(self, predicts, expls, errors):
        new_predicts = [] 
        new_expls = [] 
        new_errors = []
        for agent in self.agents:
            new_result, new_err = agent.update_answer(predicts, expls, errors)
            new_predict, new_expl = (new_result['predict'], new_result['explanation']) if not new_err else ("", "")
            new_predicts.append(new_predict)
            new_expls.append(new_expl)
            new_errors.append(new_err)
        return (new_predicts, new_expls, new_errors)