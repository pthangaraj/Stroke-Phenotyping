#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods
#This script is the gridsearch analysis for each classifier, arguments are entered into gridsearch_analysis_stroke.py
import MySQLdb
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import sys
import csv
import numpy as np
import random
import scipy as sp
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from keras import regularizers
from scipy import stats
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from scipy.sparse import csr_matrix
import pickle
import time
import datetime
from datetime import date
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
print "start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
def main(train_mat,train_labels,mod):
    X_train=sp.sparse.load_npz(train_mat)
    y_train=np.load(train_labels)
    #LR model
    if mod=="lr":
        model_s=linear_model.LogisticRegression(penalty = 'l1')
        parameters = {'C':[.001,.01,.05,.1,.5,1,5,10,100]}
    #EN model
    elif mod=="en":
        model_s=linear_model.SGDClassifier(penalty='elasticnet',loss='log')
        parameters = {'l1_ratio':[1e-10, 1e-6,.01,.1,.5,1], 'alpha':[1e-10,1e-6,.01,.1, 10,1e6,1e10]}
    #RF model
    elif mod=="rf":
        model_s= ensemble.RandomForestClassifier(max_features="sqrt", oob_score=True)
	parameters = {'n_estimators':[int(1e4)],'max_depth':[int(1e3)]}
    #AB model
    elif mod=="ab":
        model_s= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=1000)
        parameters = {'learning_rate':[.1,.5,1], 'n_estimators':[1000]}
    #GB model
    elif mod=="gb": 
        model_s=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=200, max_depth=10)
        parameters = {'learning_rate':[1e-3,.1,10], 'subsample':[1e-10,1e-6,.01,.5,1],'n_estimators':[200,500,1000],'max_depth':[10,100]}
    #NN model- not used in final analysis
#    def create_model(learn_rate=0.0001,momentum=0.5,dropout_rate=0.2):
#        model = Sequential()
#        model.add(Dense(units=64, activation='relu', kernel_regularizer=regularizers.l1(0.1),input_dim=X_new.shape[1]))
#	model.add(Dropout(dropout_rate))
#        model.add(Dense(units=2, activation='softmax'))
#	optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum,nesterov=True)
#    #model_s.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizer, metrics=['accuracy'])
#	#model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
#	return model
#    learn_rate=0.0001
#    momentum=0.75
#    dropout_rate=.4
#    model = Sequential()
#    model.add(Dense(units=64, activation='relu', kernel_regularizer=regularizers.l1(0.1),input_dim=X_new.shape[1]))
#    model.add(Dropout(dropout_rate))
#    model.add(Dense(units=2, activation='softmax'))
#    optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum,nesterov=True)
#    #model_s.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizer, metrics=['accuracy'])
#    #model_s = KerasClassifier(build_fn=create_model, epochs=5, batch_size=40, verbose=1)
#    learn_rate = [.0001,.0005,.00001]
#    momentum = [0.0,0.5,0.7]
#    optimizer = ['SGD', 'RMSprop','Adam']
#    dropout_rate = [0.2,0.3,0.5]
#    activation=['softmax','relu','sigmoid']
#    parameters = dict(learn_rate=learn_rate,momentum=momentum,dropout_rate=dropout_rate)
#    #param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model_s, param_grid=parameters,scoring=['accuracy','precision','f1'] , refit='precision',cv=10,n_jobs=20)
    grid_result=grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_f1']
    stds = grid_result.cv_results_['std_test_f1']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    means = grid_result.cv_results_['mean_test_precision']
    stds = grid_result.cv_results_['std_test_precision']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    means = grid_result.cv_results_['mean_test_accuracy']
    stds = grid_result.cv_results_['std_test_accuracy']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
if __name__=="__main__":
    args = dict([x.split('=') for x in sys.argv[1:]])
    print >> sys.stderr, ""
    print >> sys.stderr, "Running gridsearch analysis "
    print >> sys.stderr, "-----------------------------------------------------------------------------"
    valid_args = ('train_matrix', 'train_labels','mod')
    for argname in args.keys():
        if not argname in valid_args:
            raise Exception("Provided argument: `%s` is not a valid argument name." % argname)
    main(train_mat = args['train_matrix'],
        train_labels = args['train_labels'],
	mod=args['mod'])
    print "end", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
