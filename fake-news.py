from torch.autograd import Variable
import torch
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import tree as tr
import heapq
from sklearn import neighbors as nb
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

#import cPickle

import os
from scipy.io import loadmat



total_fake = 0
total_real = 0
fake_count = {}
real_count = {}

all_key_words = {}
all_words_keys = []


def add_non_overlapping(lst1, lst2):
    # add words from the other class that do not occur in this class and give it a count of 0
    for word in lst2:
        if word not in lst1:
            lst1[word] = 0
    return lst1


def count_occurrences(fileList, dictionary, total_count):
    # creates a dictionary of words paired with the number of headlines in the file it occurs in
    num_lines = len(fileList)
    count = 0
    for line in fileList[:int(num_lines*0.7)]:
        parts = line.strip().split(" ")

        for word in list(set(parts)):
            lower_word = word.strip()

            if lower_word not in all_key_words:
                all_key_words[lower_word] = count
                all_words_keys.append(lower_word)
                count += 1


            if lower_word in dictionary:
                dictionary[lower_word] += 1
            else:
                dictionary[lower_word] = 1

        total_count += 1

    return dictionary, total_count

def create_input_single(headline):
    # create an input from a single headline

    word_dict = create_headline_dictionary(headline)
    x = [0]*len(all_key_words)

    for word in word_dict:
        if word in all_key_words:
            x[all_key_words[word]] += 1

    return x

def create_headline_dictionary(headline):
    # creates a dictionary of words from a headline
    parts = headline.split(" ")
    word_dict = {}

    for word in parts:
        word = word.lower()
        if word in fake_count or word in real_count:
            word_dict[word] = word

    return word_dict

def create_input_multiple(fileList, output):
    # create a numpy array of input from a file of headlines
    x = []
    for headline in fileList:
        x.append(create_input_single(headline))

    x = np.array(x)
    y = np.array([output]*x.shape[0])

    return x, y


def create_input_all(real_lst, fake_lst):
    # given a list of fake headlines and a list of real headlines create numpy inputs
    num_lines_real = len(real_lst)
    num_lines_fake = len(fake_lst)

    x_real, y_real = create_input_multiple(real_lst[0 : int(num_lines_real*0.7)], 1)
    x_fake, y_fake = create_input_multiple(fake_lst[0 : int(num_lines_fake*0.7)], 0)

    x = np.vstack((x_real, x_fake))
    y = np.vstack((y_real.reshape(y_real.shape[0], 1), y_fake.reshape(y_fake.shape[0], 1)))


    x_val_real, y_val_real = create_input_multiple(real_lst[int(num_lines_real*0.7) : int(num_lines_real*0.85)], 1)
    x_val_fake, y_val_fake = create_input_multiple(fake_lst[int(num_lines_fake*0.7) : int(num_lines_fake*0.85)], 0)

    x_val = np.vstack((x_val_real, x_val_fake))
    y_val = np.vstack((y_val_real.reshape(y_val_real.shape[0], 1), y_val_fake.reshape(y_val_fake.shape[0], 1)))

    x_test_real, y_test_real = create_input_multiple(real_lst[int(num_lines_real*0.85) : int(num_lines_real*1)], 1)
    x_test_fake, y_test_fake = create_input_multiple(fake_lst[int(num_lines_fake*0.85) : int(num_lines_fake*1)], 0)

    x_test = np.vstack((x_test_real, x_test_fake))
    y_test = np.vstack((y_test_real.reshape(y_test_real.shape[0], 1), y_test_fake.reshape(y_test_fake.shape[0], 1)))

    return x, y, x_val, y_val, x_test, y_test



def multiple_decision_trees(real_lst, fake_lst):
    x, y, x_val, y_val, x_test, y_test = create_input_all(real_lst, fake_lst)
    y = y.reshape((-1))
    y_val = y_val.reshape((-1))
    y_test = y_test.reshape((-1))

    model1 = tr.DecisionTreeClassifier(max_depth=80)
    model2 = tr.DecisionTreeClassifier(max_depth=50)
    model3 = tr.DecisionTreeClassifier(max_depth=20)
    model1.fit(x, y)
    model2.fit(x, y)
    model3.fit(x, y)

    validation_score = test_multi_decision_tree(model1, model2, model3, x_val, y_val)
    test_score = test_multi_decision_tree(model1, model2, model3, x_test, y_test)

    print("validation score: " + str(validation_score))
    print("test score: " + str(test_score))



def test_multi_decision_tree(model1, model2, model3, x, y):
    prediction1 = model1.predict(x)
    prediction2 = model2.predict(x)
    prediction3 = model3.predict(x)

    out = (prediction1 + prediction2 + prediction3)/3.0
    correct=0
    total = 0

    for i in range(y.shape[0]):
        if out[i] > 0.5:
            result = 1
        else:
            result = 0

        if result == y[i]:
            correct += 1
        total += 1

    return float(correct)/total

def train_logistic(x, y_classes, x_val, y_val_classes, x_test, y_test_classes, iterations, weight_decay, show_graph):
    output_dim = 1
    dim_h = 60
    dim_h2 = 10
    dim_x = x.shape[1]

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, output_dim),
        torch.nn.Sigmoid()
    )

    loss_fn = torch.nn.BCELoss()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    x_axis = []
    y_axis = []
    y_training_axis = []
    y_test_axis = []

    # train the logistic regression model
    i = 50
    while i <= iterations:
        optimizer.zero_grad()  # Zero out the previous gradient computation
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to

        if i % 100 == 0:
            print "training: " + str(test_score(model(x).data.numpy(), y_classes.data.numpy()))
            print "validation: " + str(test_score(model(x_val).data.numpy(), y_val_classes.data.numpy()))
            print "test: " + str(test_score(model(x_test).data.numpy(), y_test_classes.data.numpy()))

            x_axis.append(i)
            y_axis.append(test_score(model(x_val).data.numpy(), y_val_classes.data.numpy()))
            y_training_axis.append(test_score(model(x).data.numpy(), y_classes.data.numpy()))
            y_test_axis.append(test_score(model(x_test).data.numpy(), y_test_classes.data.numpy()))

        i += 1

    if show_graph:
        plt.plot(x_axis, y_axis, 'yo-', label="validation set")
        plt.plot(x_axis, y_training_axis, 'go-', label="training set")
        plt.plot(x_axis, y_test_axis, 'bo-', label="test set")
        plt.xlabel("iteration")
        plt.ylabel("accuracy rate")
        plt.legend(loc='top left')
        plt.title('accuracy vs iteration')
        plt.savefig("part4_iteration.png")
        plt.show()


    return model

def logisticRegressionNetwork(real_lst, fake_lst):
    x, y, x_val, y_val, x_test, y_test = create_input_all(real_lst, fake_lst)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    # create the input pytorch variables
    x = Variable(torch.from_numpy(x), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(y), requires_grad=False).type(dtype_float)


    x_val = Variable(torch.from_numpy(x_val), requires_grad=False).type(dtype_float)
    y_val_classes = Variable(torch.from_numpy(y_val), requires_grad=False).type(dtype_float)

    x_test = Variable(torch.from_numpy(x_test), requires_grad=False).type(dtype_float)
    y_test_classes = Variable(torch.from_numpy(y_test), requires_grad=False).type(dtype_float)

    train_logistic(x, y_classes, x_val, y_val_classes, x_test, y_test_classes, 1000, 0.003, True)

    x_axis = []
    y_axis = []
    y_training_axis = []
    y_test_axis = []

    # check performance with different weight decays
    i = 0.0
    while i <= 0.01:
        model = train_logistic(x, y_classes, x_val, y_val_classes, x_test, y_test_classes, 400, i, False)
        x_axis.append(i)
        y_axis.append(test_score(model(x_val).data.numpy(), y_val_classes.data.numpy()))
        y_training_axis.append(test_score(model(x).data.numpy(), y_classes.data.numpy()))
        y_test_axis.append(test_score(model(x_test).data.numpy(), y_test_classes.data.numpy()))
        i+= 0.001

    plt.plot(x_axis, y_axis, 'yo-', label="validation set")
    plt.plot(x_axis, y_training_axis, 'go-', label="training set")
    plt.plot(x_axis, y_test_axis, 'bo-', label="test set")
    plt.xlabel("regularization parameter")
    plt.ylabel("accuracy rate")
    plt.legend(loc='top left')
    plt.title('accuracy per regularization')
    plt.savefig("part4_weight decay.png")
    plt.show()



    model = train_logistic(x, y_classes, x_val, y_val_classes, x_test, y_test_classes, 600, 0.003, False)

    return model


def knn(real_lst, fake_lst):
    x, y, x_val, y_val, x_test, y_test = create_input_all(real_lst, fake_lst)
    y = y.reshape((-1))
    y_val = y_val.reshape((-1))
    y_test = y_test.reshape((-1))

    x_axis = []
    y_axis_validation = []
    y_axis_test = []

    i = 1
    while i <= 21:
        clf = nb.KNeighborsClassifier(i)
        clf.fit(x, y)
        score = clf.score(x_val, y_val)
        score2 = clf.score(x_test, y_test)

        x_axis.append(i)
        y_axis_validation.append(score)
        y_axis_test.append(score2)
        print("value of k: " + str(i) + " validation score: " + str(score) + " test score: " + str(score2))
        i += 1

    plt.plot(x_axis, y_axis_validation, 'yo-', label="validation set")
    plt.plot(x_axis, y_axis_test, 'bo-', label="test set")
    plt.legend(loc='top left')
    plt.xlabel("k")
    plt.ylabel("accuracy rate")
    plt.title('accuracy vs k')
    plt.show()

def svm(real_lst, fake_lst):
    x, y, x_val, y_val, x_test, y_test = create_input_all(real_lst, fake_lst)
    y = y.reshape((-1))
    y_val = y_val.reshape((-1))
    y_test = y_test.reshape((-1))

    x_axis = []
    y_axis_validation = []
    y_axis_test = []

    clf = SVC(max_iter=70, C=0.8)
    clf.fit(x, y)

    score = clf.score(x_val, y_val)
    score2 = clf.score(x_test, y_test)


    print("validation score: " + str(score))
    print("test score: " + str(score2))

def naivebayes(real_lst, fake_lst):
    x, y, x_val, y_val, x_test, y_test = create_input_all(real_lst, fake_lst)
    y = y.reshape((-1))
    y_val = y_val.reshape((-1))
    y_test = y_test.reshape((-1))

    df = pd.read_csv("data.csv")
    X_train, X_test, y_train, y_test = train_test_split(df['Headline'], df.Label, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    fake_lst = [a.strip() for a in fake_lst]
    count_test = count_vectorizer.transform(fake_lst)
    #
    # print count_vectorizer
    clf = MultinomialNB(alpha=0.3)
    clf.fit(count_train, y_train)
    #clf.fit(x, y)

    #score = clf.score(x_val, y_val)
    score = clf.score(count_test, np.ones(len(fake_lst)))
    print("test score: " + str(score))



random.seed(786)
torch.manual_seed(6)

real_lst = open("clean_real.txt").readlines()
fake_lst = open("clean_fake.txt").readlines()
random.shuffle(real_lst)
random.shuffle(fake_lst)

real_count, total_real = count_occurrences(real_lst, real_count, total_real)
fake_count, total_fake = count_occurrences(fake_lst, fake_count, total_fake)


real_count = add_non_overlapping(real_count, fake_count)
fake_count = add_non_overlapping(fake_count, real_count)


print("knn")
#model = logisticRegressionNetwork(real_lst, fake_lst)
knn(real_lst, fake_lst)

print("svm")
svm(real_lst, fake_lst)

print("decision forest")
multiple_decision_trees(real_lst, fake_lst)

print("naive bayes")
naivebayes(real_lst, fake_lst)
