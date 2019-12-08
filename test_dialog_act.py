import nltk
from sklearn.svm import SVC
import numpy as np
from time import time
import pickle


def compose_dictionary(posts):
    result = []
    for post in posts:
        for word in nltk.word_tokenize(post.text):
            if word not in result:
                result.append(word)
    return result    


def vectorize_string(text, dictionary):
    result = np.zeros(len(dictionary))
    for word in nltk.word_tokenize(text):
        idx = dictionary.index(word)
        result[idx] = 1.0
    return result


def collect_classes(posts):
    result = []
    for post in posts:
        label = post.get('class')
        if label not in result:
            result.append(label)
    return result


def compose_training_data(posts, dictionary, all_labels):
    features = []
    labels = []
    for post in posts:
        vector = vectorize_string(post.text, dictionary)
        features.append(vector)
        
        label = post.get('class')
        labels.append(all_labels.index(label))
    return np.asarray(features), np.asarray(labels)
    
    
    
if __name__ == '__main__':
    #nltk.download('nps_chat')
    print('getting dictionary and labels')
    posts = nltk.corpus.nps_chat.xml_posts() #[:10000]
    global_dictionary = compose_dictionary(posts)
    all_labels = collect_classes(posts)
    #print('total dictionary size', len(global_dictionary))
    #print('all labels', labels)
    
    #start = time()
    #test = vectorize_string(posts[0].text, global_dictionary)
    #end = time()
    #print(test, end - start)
    
    print('compiling data')
    features, labels = compose_training_data(posts, global_dictionary, all_labels)
    
    #print(features)
    #print(labels)
    
    train_features = features[:9000]
    train_labels = labels[:9000]
    test_features = features[9000:]
    test_labels = labels[9000:]
    
    print('fitting model')
    model = SVC(gamma='auto')
    model.fit(train_features, train_labels)
    score = model.score(test_features, test_labels)
    print('score', score)
    with open('model.svc', 'wb') as outfile:
        pickle.dump((model, labels), outfile)