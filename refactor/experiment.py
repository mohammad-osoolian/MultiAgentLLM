from agent import DebateAgent
from experiment import *
from dataset import *
import datetime
import os
from agent import *
from logger import *
from dataset import *
import metrics
import numpy as np

class ZeroShotExperiment:
    def __init__(self, data:DataHandler, agent:ZeroShotLlm, basepath):
        self.data = data
        self.agent = agent
        self.basepath = basepath
        self.make_directories()
        self.agent.logger = AgentLogger(f'{self.basepath}/logs/log-agent.json')
        self.results = self.data.results_template()
        self.metrics = {}
            
    def make_directories(self):
        if os.path.exists(self.basepath):
            raise FileExistsError(f"path '{self.basepath}' already exists")
        os.mkdir(self.basepath)
        os.mkdir(f'{self.basepath}/logs/')
        os.mkdir(f'{self.basepath}/results/')
    
    def info(self):
        info = dict()
        info['agent_info'] = self.agent.info()
        info['dataset_info'] = self.data.info()
        info['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return info
    
    def run(self):
        self.write_experiment_card()
        print('Starting Experiment...\n---------------------------------')
        prediction_results = self.agent.batch_predict(self.data.texts)
        predicts, errors = zip(*prediction_results)
        self.results['error'] = errors
        predicts = pd.DataFrame(predicts)
        self.results = self.results.assign(**predicts)
        
        self.messure_metrics()
        self.write_results()
    
    def write_results(self):
        self.results.to_csv(f'{self.basepath}/result/classification-result.tsv', sep='\t', index=False)
        with open(f'{self.basepath}/metrics.json', 'w') as f:
            f.write(json.dumps(self.metrics))

    def write_experiment_card(self):
        card = self.info()
        with open(f'{self.basepath}/experiment-card.json', 'w') as f:
            f.write(json.dumps(card))

    def messure_metrics(self):
        predicts = self.results['predict'].tolist()
        labels = self.results['label'].tolist()
        # self.metrics['accuracy'] = metrics.accuracy(labels, predicts)
        self.metrics['ml_accuracy'] = metrics.multi_label_acc(labels, predicts, 28)


class DebateExperiment:
    def __init__(self, data:DataHandler, agents: list[DebateAgent], basepath):
        self.data = data
        self.agents = agents
        self.predictions = []
        self.final_predictions = []
        self.basepath = basepath
        self.make_directories()
        for i in range(len(self.agents)):
            self.agents[i].logger = AgentLogger(f'{self.basepath}/logs/agent{i+1}.json')
        self.metrics = {}

    def check_agreement(self, predicts):
        # return len(set(predicts)) == 1
        first = predicts[0]
        return all(p == first for p in predicts)

    def list2onehot(self, lst, n_labels):
    # lst = list(map(int, lst.split(',')))
        vector = np.zeros(shape=(n_labels,))
        for l in lst:
            vector[l - 1] = 1
        return vector

    def onehot2list(self, vector):
        result = []
        for i in range(len(vector)):
            if vector[i]:
                result.append(i+1)
        return result

    def final_predict(self, predicts):
        if type(predicts[0]) == list:
            final_labels = np.ones(shape=(max(max(pr) for pr in predicts),))
            for p in predicts:
                final_labels *= self.list2onehot(p, max(max(pr) for pr in predicts))
            return self.onehot2list(final_labels)
        else:
            freq = {}
            for p in predicts:
                freq[p] = freq.get(p, 0) + 1
            return max(freq, key=freq.get)
    
    def run(self):
        self.single_agent()
        self.debate()
        self.messure_metrics()
        self.write_results()

    def reformat(self, agent_results):
        predictions = []
        for i in range(len(self.data.texts)):
            sample = {'predicts':[], 'explanations':[], 'errors':[]}
            for j in range(len(self.agents)):
                result, err = agent_results[j][i]
                pred, expl = (result['predict'], result['explanation'])
                sample['errors'].append(err)
                sample['predicts'].append(pred)
                sample['explanations'].append(expl)
            predictions.append(sample)
        return predictions

    def single_agent(self):
        single_agent_results = []
        for agent in self.agents:
            agent_results = agent.batch_predict(self.data.texts)
            for i in range(len(agent_results)):
                if agent_results[i][1]:
                    agent_results[i] = ({'predict': '', 'explanation': ''}, 1)
            single_agent_results.append(agent_results)
        for i in range(len(self.agents)):
            self.write_single_agent_results(single_agent_results[i], f'{self.basepath}/results/agent-result-{i+1}.tsv')
            self.metrics[f'agent{i+1}'] = self.messure_agent_metrics(single_agent_results[i])

        self.predictions = self.reformat(single_agent_results)
    
    def write_single_agent_results(self, agent_result, path):
        rows = [{'predict': d[0]['predict'], 'explanation': d[0]['explanation'], 'err': d[1]} for d in agent_result]
        df = self.data.results_template().assign(**pd.DataFrame(rows))
        df.to_csv(path, sep='\t', index=False)      

    def debate(self):
        for i, sample in enumerate(self.predictions):
            print(self.data.texts[i])
            predicts, expls, errors = sample['predicts'], sample['explanations'], sample['errors']
            for round in range(3):
                if self.check_agreement(predicts):
                    break
                print("DEBATE")
                predicts, expls, errors = self.debate_round(predicts, expls, errors, self.data.texts[i])
                print(predicts)
            self.final_predictions.append(self.final_predict(predicts))
    
    def debate_round(self, predicts, expls, errors, text):
        new_predicts = [] 
        new_expls = [] 
        new_errors = []
        for agent_index, agent in enumerate(self.agents):
            new_result, new_err = agent.update_answer(predicts, expls, errors, text, agent_index)
            new_predict, new_expl = (new_result['predict'], new_result['explanation']) if not new_err else ("", "")
            new_predicts.append(new_predict)
            new_expls.append(new_expl)
            new_errors.append(new_err)
        return (new_predicts, new_expls, new_errors)

    def make_directories(self):
        if os.path.exists(self.basepath):
            raise FileExistsError(f"path '{self.basepath}' already exists")
        os.mkdir(self.basepath)
        os.mkdir(f'{self.basepath}/logs/')
        os.mkdir(f'{self.basepath}/results/')
    
    def messure_agent_metrics(self, agent_result):
        agent_metrics = {}
        predicts = [d[0]['predict'] for d in agent_result]
        labels = self.data.results_template()['label'].tolist()
        errors = [d[1] for d in agent_result]
        agent_metrics['accuracy'] = metrics.accuracy(labels, predicts)
        agent_metrics['ml_accuracy'] = metrics.multi_label_acc(labels, predicts, 28)
        agent_metrics['error_rate'] = metrics.error_rate(errors)
        return agent_metrics


    def messure_metrics(self):
        predicts = self.final_predictions
        labels = self.data.results_template()['label'].tolist()
        all_predictions = list(zip(*[d['predicts'] for d in self.predictions]))
        self.metrics['analysis'] = metrics.analysis_debate_potential(all_predictions, labels)
        self.metrics['accuracy'] = metrics.accuracy(labels, predicts)
    
    def write_results(self):
        df = self.data.results_template().assign(predict=self.final_predictions)
        df.to_csv(f'{self.basepath}/results/debate-result.tsv', sep='\t', index=False)
        with open(f'{self.basepath}/metrics.json', 'w') as f:
            f.write(json.dumps(self.metrics))
