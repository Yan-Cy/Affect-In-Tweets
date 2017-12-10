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


def load_data_and_labels(data_dir, task, set_label):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    
    if set_label == 'train':
        train_data = list(open('./datasets/EI-oc-En-train/EI-oc-En-{}-train.txt'.format(task), "r").readlines())
        train_data = [s.strip().split('\t') for s in train_data]
        data = train_data
    elif set_label == 'test':
        data = list(open('./datasets/2018-EI-oc-En-dev/2018-EI-oc-En-{}-dev.txt'.format(task), "r").readlines())
        data = [s.strip().split('\t') for s in data[1:]]
    else:
        print 'Wrong set label !'
        return None  

    # Split by words
    x_text = [clean_str(sent[1]).split(' ') for sent in data]
    y = np.zeros((len(x_text), 4))
    for i, sent in enumerate(data):
        y[i, int(sent[-1][0])] = 1
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def build_vocabulary(sentences):
    #model = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    assert(type(sentences[0]) == list)
    model_name = '../wordvec/gensim_vectors.txt'
    #model_name = '../wordvec/GoogleNews-vectors-negative300.bin'
    print 'loading Word Vector model from', model_name
    model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=False)

    vector_size = len(model.wv.syn0[0])
    print 'Word Vector size: ', vector_size

    #model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4) 
    #print 'Vocabulary Size:', len(model.wv.syn0)
    max_document_length = 30

    x = np.zeros((len(sentences), max_document_length))
    match_ratio = 0.0
    for i, sentence in enumerate(sentences):
        num_word = 0.0
        for j, word in enumerate(sentence):
            if j >= max_document_length:
                break
            #print type(word)
            if word in model.wv.vocab:
                num_word += 1
                x[i,j] = model.wv.vocab[word].index
        match_ratio += num_word / len(sentence)
    print 'Average sentence matching ratio', match_ratio / len(sentences)
    return x, model.wv.syn0
