#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods
#This script is to build multi-processor robustness analysis 10-fold cross validation, 10 different percentages (.001-1) of case-control trianing sets for Supplementary figures 1 and 2 to run robust_single_sample.py
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
training_set_matrix_filename=
training_set_labels_filename=
testing_set_labels_filename=
testing_set_matrix_filename=
sample_inds_filename=

print "start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#open training sparse matrix and training labels
filename_mtrain=training_set_matrix_filename + case + control + '.npz'
matrix_train=sp.sparse.load_npz(filename_mtrain)
filename_trlabels=training_set_labels_filename + case + control + '.npy'
train_labels=np.load(filename_trlabels)

#open test sparse matrix and test labels
filename_telabels=testing_set_labels_filename+case+control+'.npy'
test_labels=np.load(filename_telabels)
filename_mtest=testing_set_matrix_filename+case+control+'.npz'
matrix_test=sp.sparse.load_npz(filename_mtest)
print matrix_train.shape,matrix_test.shape
#10-fold cross validation of each model,saving each of the 10 folds,AUCROC and AUCPR by cross validation, and testing the same test set on each of the folds, AUC-ROC and AUC-PR by each of the 10 trianing models on the test set 
#open mrn2label
##Robustness analysis to determine the minimum number of patients needed to train stroke model
#Remove 10% of patients at a time, random sampling with replacement 10x

len_inds=matrix_train.shape[0]
inds=np.arange(len_inds)
print len_inds, len(inds)
matrix=sp.sparse.load_npz(filename_mtrain)
selected_params=[]
#repeat robustness algorithm 10 times
for i in range(0,10):
#make common training sample
#remove 10% of data everytime
    sample_inds=inds
    pct=[1,.9,.7,.5,.3,.1,.05,.01,.005,.001]
    for j in range(0,10):
	sample_inds=random.sample(sample_inds,int(len(inds)*pct[j]))
	filename_sampinds=sample_inds_filename+str(i)+str(pct[j])+case+control+".npy"
        np.save(filename_sampinds,sample_inds)
	cmd='''bsub -o cv_%d_%s%s%s.out -e cv_%d_%s%s%s.err -M 5000000 "python robust_single_sample.py train_matrix=%s test_matrix=%s train_labels=%s test_labels=%s sample_inds=%s rep=%d sz=%s case=%s control=%s"''' % (i,str(pct[j]),case,control,i,str(pct[j]),case,control,filename_mtrain,filename_mtest,filename_trlabels,filename_telabels,filename_sampinds,i,str(pct[j]),case,control)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()




