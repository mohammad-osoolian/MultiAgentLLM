from agent import DebateAgent
from experiment import *
from dataset import *
import datetime
import os
from agent import *
from logger import *
from dataset import *
import metrics

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
        self.metrics['accuracy'] = metrics.accuracy(labels, predicts)


class DebateExperiment:
    def __init__(self, data:DataHandler, agents: list[DebateAgent], basepath):
        self.data = data
        self.agents = agents
        self.basepath = basepath
        self.make_directories()
        for i in range(len(self.agents)):
            self.agents[i].logger = AgentLogger(f'{self.basepath}/logs/agent{i+1}.json')
        self.metrics = {}
        self.results = self.data.results_template()

    def check_agreement(self, predicts):
        return len(set(predicts)) == 1

    def final_predict(self, predicts):
        freq = {}
        for p in predicts:
            freq[p] = freq.get(p, 0) + 1
        return max(freq, key=freq.get)
    
    def run(self):
        self.single_agent()
        self.write_single_agent_results()
        self.messure_agent_metrics()
        self.debate()
        self.messure_metrics()
        self.write_results()


    def single_agent(self):
        for index, agent in enumerate(self.agents):
            agent_results = agent.batch_predict(self.data.texts)
            for i in range(len(agent_results)):
                if agent_results[i][1]:
                    agent_results[i] = ({'predict': '', 'explanation': ''}, 1)
            rows = [{f'predict{index+1}': d[0]['predict'], f'explanation{index+1}': d[0]['explanation'], f'error{index+1}': d[1]} for d in agent_results]
            self.results = self.results.assign(**pd.DataFrame(rows))

    
    def write_single_agent_results(self):
        for i in range(len(self.agents)):
            df = self.results[['id', 'label', f'predict{i+1}', f'explanation{i+1}', f'error{i+1}']]
            df.to_csv(f'{self.basepath}/results/agent-result-{i+1}.tsv', sep='\t', index=False)      

    def debate(self):
        debate_predicts = []
        for i, row in self.results.iterrows():
            for agent in self.agents:
                agent.clean()

            predicts = [row[f'predict{i+1}'] for i in range(len(self.agents))]
            expls = [row[f'explanation{i+1}'] for i in range(len(self.agents))]
            errors = [row[f'error{i+1}'] for i in range(len(self.agents))]

            for round in range(3):
                if self.check_agreement(predicts):
                    break
                print("DEBATE")
                predicts, expls, errors = self.debate_round(predicts, expls, errors, self.data.texts[i])
                print(predicts)
            debate_predicts.append(self.final_predict(predicts))
        self.results['debate_predict'] = debate_predicts

    
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
    
    def messure_agent_metrics(self):
        labels = self.results['label'].tolist()
        for i in range(len(self.agents)):
            agent_metrics = {}
            predicts = self.results[f'predict{i+1}'].tolist()
            errors = self.results[f'error{i+1}'].tolist()
            agent_metrics['accuracy'] = metrics.accuracy(labels, predicts)
            agent_metrics['error_rate'] = metrics.error_rate(errors)
            self.metrics[f'agent{i+1}'] = agent_metrics


    def messure_metrics(self):
        predicts = self.results['debate_predict'].tolist()
        labels = self.results['label'].tolist()
        all_predictions = [self.results[f'predict{i+1}'].tolist() for i in range(len(self.agents))]
        self.metrics['analysis'] = metrics.analysis_debate_potential(all_predictions, labels)
        self.metrics['accuracy'] = metrics.accuracy(labels, predicts)
    
    def write_results(self):
        df = self.results[['id', 'label', 'debate_predict']]
        df.to_csv(f'{self.basepath}/results/debate-result.tsv', sep='\t', index=False)
        with open(f'{self.basepath}/metrics.json', 'w') as f:
            f.write(json.dumps(self.metrics))
