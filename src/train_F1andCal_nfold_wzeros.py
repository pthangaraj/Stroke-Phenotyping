#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods
#This script is to build cross validation scores for Supplementary Figures 5-19 and calculate max f1, precision, recall, and thresholds for Supplementary Table 6
import MySQLdb
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import sys
import csv
import numpy as np
import random
import scipy as sp
import subprocess
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
from sklearn.metrics import precision_recall_curve
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
import statsmodels.api as sm
import statsmodels.tools.tools
case=sys.argv[1]
control=sys.argv[2]
training_set_matrix_filename=
training_set_labels_filename=
testing_set_matrix_filename=
testing_set_labels_filename=
cross_val_training_set_probabilities_filename=
cross_val_testing_set_probabilities_filename=
cross_val_maxthreshold_bybeta_filename=
cross_val_maxF1_bybeta_filename

#get training matrix
filename_mtrain=training_set_matrix_filename + case + control + '.npz'
matrix_train=sp.sparse.load_npz(filename_mtrain)
filename_labels=training_set_labels_filename + case + control + '.npy'
train_labels=np.load(filename_labels)
filename_mtest=testing_set_matrix_filename + case + control + '.npz'
matrix_test=sp.sparse.load_npz(filename_mtest)
filename_labels=testing_set_labels_filename + case + control + '.npy'
test_labels=np.load(filename_labels)
C_val=.1
##divide training matrix into folds for F1:
n=10
k=50
cv = model_selection.StratifiedKFold(n_splits=n)
cv_split=cv.split(matrix_train,train_labels)
##cv_tr_max_thresh rows:max_thresh for model type  cols:folds
cv_tr_max_f1_100=np.zeros(shape=(5,n,4))
cv_tr_max_thresh_100=np.zeros(shape=(5,n,4))
cv_tr_probs=np.zeros((len(train_labels),7))
#probabilities from holdout test set
cv_te_probs=np.zeros((5,len(test_labels),n))
#number of bins and thresholds
ivls=100
thr=.1
max_ivl=int(ivls*(1-thr)+1)
total_labels=np.zeros((len(train_labels)))
last=0
print "start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
for i, (train,test) in enumerate(cv_split):
    cases=[]
    controls=[]
    for l in range(0,len(train_labels[test])):
        if train_labels[test][l]==1:
            cases.append(l)
        else:
            controls.append(l)
    lr_clf_s=linear_model.LogisticRegression(penalty = 'l1', C = C_val)
    ##RF model
    rf_clf_s= ensemble.RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, oob_score=True)
    ##AB model
    ab_clf_s= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=1000)
   ## #GB model 
    gb_clf_s=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=1000, max_features="sqrt", max_depth=10)
    #EN model
    en_clf_s=linear_model.SGDClassifier(penalty='elasticnet',l1_ratio=.01,alpha=.01,loss='log')
##di#vide while mainintaining 1 to 1 ratio
##tr#ain the models on the folds
##ge#t the probabilities from each fold
    lr_clf_s.fit(matrix_train[train],train_labels[train])
    lr_test_label=lr_clf_s.predict_proba(matrix_train[test])
    lr_test_label_ho=lr_clf_s.predict_proba(matrix_test)
    cv_te_probs[0,:,i]=lr_test_label_ho[:,1]
    rf_clf_s.fit(matrix_train[train],train_labels[train])
    rf_test_label=rf_clf_s.predict_proba(matrix_train[test])
    rf_test_label_ho=rf_clf_s.predict_proba(matrix_test)
    cv_te_probs[1,:,i]=rf_test_label_ho[:,1]
    ab_clf_s.fit(matrix_train[train],train_labels[train])
    ab_test_label=ab_clf_s.predict_proba(matrix_train[test])
    ab_test_label_ho=ab_clf_s.predict_proba(matrix_test)
    cv_te_probs[2,:,i]=ab_test_label_ho[:,1]
    gb_clf_s.fit(matrix_train[train],train_labels[train])
    gb_test_label=gb_clf_s.predict_proba(matrix_train[test])
    gb_test_label_ho=gb_clf_s.predict_proba(matrix_test)
    cv_te_probs[3,:,i]=gb_test_label_ho[:,1]
    en_clf_s.fit(matrix_train[train],train_labels[train])
    en_test_label=en_clf_s.predict_proba(matrix_train[test])
    en_test_label_ho=en_clf_s.predict_proba(matrix_test)
    cv_te_probs[4,:,i]=en_test_label_ho[:,1]
#  log the probabilities from each fold
    cv_tr_probs[last:last+len(train_labels[test]),0]=lr_test_label[:,1]
    cv_tr_probs[last:last+len(train_labels[test]),1]=rf_test_label[:,1]
    cv_tr_probs[last:last+len(train_labels[test]),2]=ab_test_label[:,1]
    cv_tr_probs[last:last+len(train_labels[test]),3]=gb_test_label[:,1]
    cv_tr_probs[last:last+len(train_labels[test]),4]=en_test_label[:,1]
    cv_tr_probs[last:last+len(train_labels[test]),5]=train_labels[test]
    cv_tr_probs[last:last+len(train_labels[test]),6]=np.full((len(train_labels[test]),),i)
    total_labels[last:last+len(train_labels[test])]=train_labels[test]
    last=last+len(train_labels[test])
#for each calibrated model, bin every 5%
    controls_100=np.random.choice(controls,len(cases)*100)
    cv_tr_probs_100=np.zeros((len(np.hstack((controls_100,cases))),5))
    cv_tr_probs_100[:,0]=lr_test_label[np.hstack((controls_100,cases)),1]
    cv_tr_probs_100[:,1]=rf_test_label[np.hstack((controls_100,cases)),1]
    cv_tr_probs_100[:,2]=ab_test_label[np.hstack((controls_100,cases)),1]
    cv_tr_probs_100[:,3]=gb_test_label[np.hstack((controls_100,cases)),1]
    cv_tr_probs_100[:,4]=en_test_label[np.hstack((controls_100,cases)),1]
    train_labels_100=train_labels[test][np.hstack((controls_100,cases))]
    print "fitdone", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    controls_100=np.random.choice(controls,len(cases)*100)
    cv_tr_probs_100=np.zeros((len(np.hstack((controls_100,cases))),5))
    cv_tr_probs_100[:,0]=lr_test_label[np.hstack((controls_100,cases)),1]
    cv_tr_probs_100[:,1]=rf_test_label[np.hstack((controls_100,cases)),1]
    cv_tr_probs_100[:,2]=ab_test_label[np.hstack((controls_100,cases)),1]
    cv_tr_probs_100[:,3]=gb_test_label[np.hstack((controls_100,cases)),1]
    cv_tr_probs_100[:,4]=en_test_label[np.hstack((controls_100,cases)),1]
    train_labels_100=train_labels[test][np.hstack((controls_100,cases))]
    #   # #calculate auroc, P, R, F1, threshold
    b=[1,1/2,1/4,1/8]
    for beta in range(0,len(b)):
        precision_lr, recall_lr, threshold_lr = precision_recall_curve(train_labels_100,cv_tr_probs_100[:,0])
        f1_lr=(1+b[beta]**2)*(precision_lr*recall_lr)/(b[beta]**2*precision_lr+recall_lr)
        precision_rf, recall_rf, threshold_rf = precision_recall_curve(train_labels_100,cv_tr_probs_100[:,1])
        f1_rf=(1+b[beta]**2)*(precision_rf*recall_rf)/(b[beta]**2*precision_rf+recall_rf)
        precision_ab, recall_ab, threshold_ab = precision_recall_curve(train_labels_100,cv_tr_probs_100[:,2])
        f1_ab=(1+b[beta]**2)*(precision_ab*recall_ab)/(b[beta]**2*precision_ab+recall_ab)
        precision_gb, recall_gb, threshold_gb = precision_recall_curve(train_labels_100,cv_tr_probs_100[:,3])
        f1_gb=(1+b[beta]**2)*(precision_gb*recall_gb)/(b[beta]**2*precision_gb+recall_gb)
        precision_en, recall_en, threshold_en = precision_recall_curve(train_labels_100,cv_tr_probs_100[:,4])
        f1_en=(1+b[beta]**2)*(precision_en*recall_en)/(b[beta]**2*precision_en+recall_en)
        cv_tr_max_f1_100[0,i,beta]=np.nanmax(f1_lr)
        cv_tr_max_f1_100[1,i,beta]=np.nanmax(f1_rf)
        cv_tr_max_f1_100[2,i,beta]=np.nanmax(f1_ab)
        cv_tr_max_f1_100[3,i,beta]=np.nanmax(f1_gb)
        cv_tr_max_f1_100[4,i,beta]=np.nanmax(f1_en)
        cv_tr_max_thresh_100[0,i,beta]=threshold_lr[np.nanargmax(f1_lr)]
        cv_tr_max_thresh_100[1,i,beta]=threshold_rf[np.nanargmax(f1_rf)]
        cv_tr_max_thresh_100[2,i,beta]=threshold_ab[np.nanargmax(f1_ab)]
        cv_tr_max_thresh_100[3,i,beta]=threshold_gb[np.nanargmax(f1_gb)]
        cv_tr_max_thresh_100[4,i,beta]=threshold_en[np.nanargmax(f1_en)]
filename=cross_val_training_set_probabilities_filename+case+control+".npy"
np.save(filename,cv_tr_probs)
filename_thresh_100=cross_val_maxthreshold_bybeta_filename+case+control+".npy"
filename_f1_100=cross_val_maxf1_bybeta_filename+case+control+".npy"
filename_te=cross_val_testing_set_probabilities_filename+case+control+".npy"
np.save(filename_thresh_100,cv_tr_max_thresh_100)
np.save(filename_f1_100,cv_tr_max_f1_100)
np.save(filename_te,cv_te_probs)
