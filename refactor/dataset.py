import pandas as pd

class DataHandler:
    def __init__(self, datapath):
        self.df = pd.read_csv(datapath, sep='\t',  dtype={'label': str})
        self.datapath = datapath
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()

    def info(self):
        info = dict()
        info['dataset_path'] = self.datapath
        info['number_of_samples'] = len(self.texts)
        return info
    
    def results_template(self) -> pd.DataFrame:
        return self.df[['id', 'label']].copy()

class ResultHandler:
    def __init__(self, datapath):
        self.df = pd.read_csv(datapath, sep='\t', dtype={'predict': str, 'label': str})
        self.labels = self.df['label'].tolist()
        self.predicts = self.df['predict'].tolist()
        self.errors = self.df['error'].tolist()
        self.ids = self.df['id'].tolist()

        