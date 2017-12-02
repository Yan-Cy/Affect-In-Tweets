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


def load_data_and_labels(data_dir):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    """
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    """

    # Load data from files
    #joy = list(open(data_dir + '/joy.txt', "r").readlines())
    #joy = [s.strip() for s in joy]
    #sadness = list(open(data_dir + '/sadness.txt', "r").readlines())
    #sadness = [s.strip() for s in sadness]
    
    anger = list(open(data_dir + '/anger-ratings-0to1.train.txt', "r").readlines())
    anger = [s.strip().split('\t') for s in anger]
    anger_dev = list(open(data_dir + '/anger-ratings-0to1.dev.gold.txt', "r").readlines())
    anger = anger + [s.strip().split('\t') for s in anger_dev]
    anger_2018 = list(open(data_dir + '/EI-reg-English-Train/EI-reg-en_anger_train.txt', "r").readlines())
    anger = anger + [s.strip().split('\t') for s in anger_2018] 
    #anger_test = list(open(data_dir + '/anger-ratings-0to1.test.target.txt', "r").readlines())
    #anger = [s.strip().split('\t') for s in anger_test]
 
    #fear = list(open(data_dir + '/fear.txt', "r").readlines())
    #fear = [s.strip() for s in fear] 

    # Split by words
    x_text = [clean_str(sent[1]) for sent in anger]
    y = np.array([[float(sent[-1])] for sent in anger])
    #y = np.array([[0] for sent in anger])

    # Generate labels
    #joy_labels = [[1,0,0,0] for _ in joy]
    #sadness_labels = [[0,1,0,0] for _ in sadness]
    #anger_labels = [[0,0,1,0] for _ in anger]
    #fear_labels = [[0,0,0,1] for _ in fear]

    #y = np.concatenate([joy_labels, sadness_labels, anger_labels, fear_labels], 0)
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
    model = gensim.models.KeyedVectors.load_word2vec_format('../wordvec/gensim_glove_vectors.txt', binary=False)
    #model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4) 
    #print 'Vocabulary Size:', len(model.wv.syn0)
    max_document_length = 30

    x = np.zeros((len(sentences), max_document_length))
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if j >= max_document_length:
                break
            #print type(word)
            if word in model.wv.vocab:
                x[i,j] = model.wv.vocab[word].index

    return x, model.wv.syn0
