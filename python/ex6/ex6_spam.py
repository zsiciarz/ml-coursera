from __future__ import print_function

import csv
import re

import numpy as np
from scipy.io import loadmat
from stemming.porter2 import stem


def get_vocabulary():
    vocabulary = {}
    with open('../../octave/mlclass-ex6/vocab.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t', skipinitialspace=True)
        for row in reader:
            vocabulary[row[1]] = int(row[0])
    return vocabulary


def process_email(email_contents, vocabulary):
    email_contents = email_contents.lower()
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)
    email_contents = re.sub(r'\d+', 'number', email_contents)
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)
    print('==== Processed Email ====')
    words = [stem(word) for word in re.findall(r"\w+", email_contents)]
    print(' '.join(words))
    word_indices = [vocabulary[word] for word in words if word in vocabulary]
    print('=========================')
    return np.array(word_indices)


def email_features(word_indices, vocabulary):
    features = np.zeros(len(vocabulary))
    for index in word_indices:
        features[index] = 1.0
    return features


if __name__ == '__main__':
    file_contents = open('../../octave/mlclass-ex6/emailSample1.txt', 'r').read()
    vocabulary = get_vocabulary()
    word_indices = process_email(file_contents, vocabulary)
    print('Word indices:\n%s' % word_indices)
    features = email_features(word_indices, vocabulary)
    print('Length of feature vector: %d' % len(features))
    print('Number of non-zero entries: %d' % sum(features > 0))
    # train SVM for spam classification
    data = loadmat('../../octave/mlclass-ex6/spamTrain.mat')
    X = np.require(data['X'], dtype=np.float64, requirements='C_CONTIGUOUS')
    y = np.require(data['y'].flatten(), dtype=np.float64)
