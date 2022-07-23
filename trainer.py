"""
In case we need to retrain the model run trainer.py and the model will be saved in project directory
"""
from trainer_utils import *

data = extract_data()
words, labels, docs_x, docs_y = intents_prepare(data)
training, output = input_prepare(words, labels, docs_x, docs_y)
net = net_setting(len(training[0]), len(output[0]))
model = train_model(training, output, net)




