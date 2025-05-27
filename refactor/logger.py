import datetime
import os
import json
import threading


class AgentLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        if os.path.exists(filepath):
            raise FileExistsError(f"file '{filepath}' already exists")
        self.lock = threading.Lock()
    
    def log_messages(self, messages):
        with self.lock:
            with open(self.filepath, 'a') as f:
                for message in messages:
                    f.write(json.dumps(message))
                    f.write('\n')


class NullLogger(AgentLogger):
    def __init__(self):
        pass
    def log_messages(self, messages):
        pass