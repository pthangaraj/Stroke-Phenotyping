#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods
#This script sets up the multiprocessing for gridsearch analysis for each classifier, takes in case and control type
import MySQLdb
from collections import defaultdict
import sys
import csv
import numpy as np
import random
import scipy as sp
import subprocess
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
import operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
case=sys.argv[1]
control=sys.argv[2]
print "start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#open training sparse matrix and training labels
filename_mtrain={training_set matrix filename} +  case + control + '.npz'
matrix_train=sp.sparse.load_npz(filename_mtrain)
filename_trlabels={training_set labels filename} + case + control + '.npy'
train_labels=np.load(filename_trlabels)

#open test sparse matrix and test labels
#10-fold cross validation of each model,saving each of the 10 folds,AUCROC and AUCPR by cross validation, and testing the same test set on each of the folds, AUC-ROC and AUC-PR by each of the 10 trianing models on the test set 
#open mrn2label
##Robustness analysis to determine the minimum number of patients needed to train stroke model
#Remove 10% of patients at a time, random sampling with replacement 10x

matrix=sp.sparse.load_npz(filename_mtrain)
selected_params=[]
#repeat robustness algorithm 10 times
pct=['lr','en','rf','ab','gb']
for i in pct:
#make common training sample
#remove 10% of data everytime
    cmd='''bsub -o grid_fixcv%s%s%s.out -e grid_fixcv%s%s%s.err -M 5000000 "python gridsearch_single_sample.py train_matrix=%s train_labels=%s  mod=%s"''' % (i,case,control,i,case,control,filename_mtrain,filename_trlabels,i)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()




