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
        os.mkdir(f'{self.basepath}/result/')
    
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

class DebateExperimet:
    def __init__(self, data, agents, basepath):
        self.data = data
        self.agents = agents
        self.basepath = basepath
        self.make_directories()
        