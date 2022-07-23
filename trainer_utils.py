import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow
import json
import numpy as np

stemmer = LancasterStemmer()


def extract_data():
    with open("intents.json", encoding='utf-8') as file:
        data = json.load(file)
    return data

#prepare data
def intents_prepare(data):
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)
    return words, labels, docs_x, docs_y

#create training and output for model
def input_prepare(words, labels, docs_x, docs_y):
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    return training, output


def net_setting(training_len, output_len):
    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, training_len])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, output_len, activation="softmax")
    net = tflearn.regression(net)
    return net


def load_model(training, output):
    net = net_setting(len(training[0]), len(output[0]))
    model = tflearn.DNN(net)
    model.load("./model.tflearn")
    return model


def train_model(training, output, net):
    model = tflearn.DNN(net)
    model.fit(training, output,  n_epoch=150, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    return model


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)
