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
        self.results.to_csv(f'{self.basepath}/results/zeroshot-result.tsv', sep='\t', index=False)
        with open(f'{self.basepath}/metrics.json', 'w') as f:
            f.write(json.dumps(self.metrics))

    def write_experiment_card(self):
        card = self.info()
        with open(f'{self.basepath}/experiment-card.json', 'w') as f:
            f.write(json.dumps(card))

    def messure_metrics(self):
        predicts = self.results['predict'].tolist()
        errors = self.results['error'].tolist()
        labels = self.results['label'].tolist()
        self.metrics['accuracy'] = metrics.accuracy(labels, predicts)
        self.metrics['error_rate'] = metrics.error_rate(errors)
        # self.metrics['ml_accuracy'] = metrics.multi_label_acc(labels, predicts, 28)


class DebateExperiment:
    def __init__(self, data:DataHandler, agents: list[DebateAgent], basepath):
        self.data = data
        self.agents = agents
        self.basepath = basepath
        self.make_directories()
        for i in range(len(self.agents)):
            self.agents[i].logger = AgentLogger(f'{self.basepath}/logs/agent{i+1}.json')
        self.metrics = {'agent_wins':[0 for i in range(len(self.agents))], 'switch_answers': 0}
        self.results = self.data.results_template()
        newcols = [f'predict{i+1}' for i in range(len(self.agents))] + [f'error{i+1}' for i in range(len(self.agents))] + [f'explanation{i+1}' for i in range(len(self.agents))]+ ['debate_predict', 'rounds']
        self.results = self.results.assign(**{col: None for col in newcols})


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
        self.debate()

    
    def write_single_agent_results(self):
        for i in range(len(self.agents)):
            df = self.results[['id', 'label', f'predict{i+1}', f'explanation{i+1}', f'error{i+1}']]
            df.to_csv(f'{self.basepath}/results/agent-result-{i+1}.tsv', sep='\t', index=False)      

    def debate(self):
        for i, row in self.data.df.iterrows():
            text = row['text']
            predicts = []
            errors = []
            expls = []
            for agent in self.agents:
                agent.clean()
            print("ID", row['id'])
            for index, agent in enumerate(self.agents):
                response, error = agent.argue(text)
                predict, expl = response['predict'], response['explanation']
                predicts.append(predict)
                errors.append(error)
                expls.append(expl)
                self.results.at[i, f'predict{index+1}'] = predict
                self.results.at[i, f'error{index+1}'] = error
                self.results.at[i, f'explanation{index+1}'] = expl

            if self.check_agreement(predicts):
                print("FIRST AGREEMENT", predicts)  
                self.results.at[i, 'debate_predict'] = self.final_predict(predicts)
                self.results.at[i, 'rounds'] = 0
                continue

            print("DEBATE", predicts)
            for round in range(3):
                prev_predicts = predicts
                predicts, expls, errors = self.debate_round(predicts, expls, errors, text)
                print(f"ROUND{round+1} --->", predicts)
                if self.check_agreement(predicts):
                    self.find_winners(prev_predicts, predicts)
                    self.results.at[i, 'rounds'] = round+1
                    break
                if self.switch_answers(prev_predicts, predicts):
                    self.metrics['switch_answers'] += 1
                    print("switch:", self.metrics['switch_answers'])
                

            self.results.at[i, 'debate_predict'] = self.final_predict(predicts)
            self.messure_metrics()
            self.write_single_agent_results()
            self.messure_agent_metrics()
            self.write_results()

    def switch_answers(self, prev, new):
        for i in range(len(prev)):
            if not(new[i] != prev[i] and new[i] in prev):
                return False
        return True

    def find_winners(self, prev_predicts, newpredicts):
        final = self.final_predict(newpredicts)
        for i in range(len(prev_predicts)):
            if prev_predicts[i] == final:
                self.metrics['agent_wins'][i] += 1
        print(self.metrics['agent_wins'])
        return
    
    def debate_round(self, predicts, expls, errors, text):
        new_predicts = [] 
        new_expls = [] 
        new_errors = []
        for agent_index, agent in enumerate(self.agents):
            # if agent_index == 0:
            #     new_predicts.append(predicts[0])
            #     new_expls.append(expls[0])
            #     new_errors.append(errors[0])
            #     continue
            new_result, new_err = agent.update_answer(predicts, expls, errors, text, agent_index)
            new_predict, new_expl = new_result['predict'], new_result['explanation']
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
    
    def messure_agent_metrics(self):
        labels = self.results['label'].tolist()
        for i in range(len(self.agents)):
            agent_metrics = {}
            predicts = self.results[f'predict{i+1}'].tolist()
            errors = self.results[f'error{i+1}'].tolist()
            agent_metrics['accuracy'] = metrics.accuracy(labels, predicts)
            agent_metrics['error_rate'] = metrics.error_rate(errors)
            # agent_metrics['ml_accuracy'] = metrics.multi_label_acc(labels, predicts, 28)
            self.metrics[f'agent{i+1}'] = agent_metrics


    def messure_metrics(self):
        predicts = self.results['debate_predict'].tolist()
        labels = self.results['label'].tolist()
        # all_predictions = [self.results[f'predict{i+1}'].tolist() for i in range(len(self.agents))]
        # self.metrics['analysis'] = metrics.analysis_debate_potential(all_predictions, labels)
        self.metrics['accuracy'] = metrics.accuracy(labels, predicts)
        # self.metrics['ml_accuracy'] = metrics.multi_label_acc(labels, predicts, 28)
    
    def write_results(self):
        df = self.results[['id', 'label', 'debate_predict', 'rounds']]
        df.to_csv(f'{self.basepath}/results/debate-result.tsv', sep='\t', index=False)
        with open(f'{self.basepath}/metrics.json', 'w') as f:
            f.write(json.dumps(self.metrics))
