import numpy as np
import re
import itertools
from collections import Counter
import gensim


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_dataset(set_label, task, gensim_model = None):
    if set_label == 'train':
        train_data = list(open('../datasets/{}-ratings-0to1.train.txt'.format(task), "r").readlines())
        train_data = [s.strip().split('\t') for s in train_data]
        dev_data = list(open('../datasets/{}-ratings-0to1.dev.gold.txt'.format(task), "r").readlines())
        dev_data = [s.strip().split('\t') for s in dev_data]
        #train_2018 = list(open('../datasets/EI-reg-English-Train/EI-reg-en_{}_train.txt'.format(task), "r").readlines())
        #train_2018 = [s.strip().split('\t') for s in train_2018]
        data = train_data + dev_data# + train_2018
    elif set_label == 'test':
        data = list(open('../datasets/{}-ratings-0to1.test.target.txt'.format(task), "r").readlines())
        data = [s.strip().split('\t') for s in data]
    else:
        print 'Wrong set label !'
        return None

    x_raw = np.array([sent[1] for sent in data])
    sentences = np.array([clean_str(sent[1]).split(' ') for sent in data])
    x, gensim_model = sentences2vectors(sentences, gensim_model)
    x_id = np.array([sent[0] for sent in data])
    if data[0][3] == 'NONE':
        y = np.array([0 for sent in data])
    else:
        y = np.array([float(sent[3]) for sent in data])
    return x, y, x_id, x_raw, gensim_model


def sentences2vectors(sentences, model = None):
    if model == None:
        #model_name = '../wordvec/gensim_glove_vectors_200.txt'
        model_name = '../wordvec/GoogleNews-vectors-negative300.bin'
        print 'loading Word Vector model from', model_name
        #model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=False)
        model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)

    vector_size = len(model.wv.syn0[0])
    vectors = np.zeros((len(sentences), vector_size))
    print 'Word Vector size: ', vector_size
    
    match_ratio = 0.0
    for i, sentence in enumerate(sentences):
        #print sentence, len(sentence)
        num_word = 0.0
        for word in sentence:
            if word in model.wv.vocab:
                num_word += 1
                vectors[i] += model.wv[word]
        vectors[i] /= num_word
        match_ratio += num_word / len(sentence)
    print 'Average sentence matching ratio', match_ratio / len(sentences)

    return vectors, model

