import pickle
import nltk
from soc_functions import *


freq_sec = 1  # frequency of running in seconds
model_file = 'nltk_naive_bayes.clf'
service = 'dialog_act'
class_taxonomy = 'thought.evaluation.speech.dialog_act'


def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains(%s)' % word.lower()] = True
    return features


def train_model():
    print('training model')
    posts = nltk.corpus.nps_chat.xml_posts()[:10000]
    featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print('accuracy:', nltk.classify.accuracy(classifier, test_set))
    return classifier


def save_model(model):
    with open(model_file, 'wb') as outfile:
        pickle.dump(model, outfile)


def load_model():
    with open(model_file, 'rb') as infile:
        model = pickle.load(infile)
    return model


def quick_inference(model, sentence):
    features = dialogue_act_features(sentence)
    return model.classify(sentence)


def load_or_train_model():
    try:
        return load_model()
    except:
        return train_model()
    

if __name__ == '__main__':
    model = load_or_train_model()
    print(quick_inference(model, 'this is only a test'))
    exit(4)
    end_time = int(time()) - freq_sec
    while True:
        start_time = end_time  # set next start time to last end time (like an inchworm)
        end_time = int(time())  # set end-time to whatever NOW is
        data = fetch_soc_range(start_time, end_time)
        
        # filter messages
        data = [i for i in data if 'sense' in i['class']]
        data = [i for i in data if 'speech' in i['class'] or 'chat' in i['class']]
        
        # classify and send messages
        
        # wait a second
        sleep(freq_sec)