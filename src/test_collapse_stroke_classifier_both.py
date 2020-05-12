#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods" 
#This script makes the testing set matrix with collapsed features, takes in case, control, and CCS level
import numpy as np
import os
import csv
import sys
import MySQLdb
import scipy.stats as stats
from collections import defaultdict
import scipy as sp
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
e2e={}
case=sys.argv[1]
control=sys.argv[2]
model=sys.argv[3]
#model2=sys.argv[4]
print "new loop start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print "first entrance to mysql", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
def credential():
        '''import login and passwrd from credential text file'''
        reader=csv.reader(open('mycnf.csv'),delimiter = ",")
        for login, password in reader:
                        login=login
                        passwd=password

        return login, passwd

login,passwd=credential()
print "first entrance to mysql", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
e2e_file={events to uniform events filename}+case+control+'.npy'
e2e=np.load(e2e_file)
e2e=e2e[()]
drug_file={drug event filename}+ case+ control + '.npy'
drug_events=np.load(drug_file)
drug_events=drug_events.tolist()
e2i_file={events2cols filename}+case+control+'.npy'
e2i=np.load(e2i_file)
e2i=e2i[()]
matrix_file={test matrix filename}+case+control+'.npz'
matrix=sp.sparse.load_npz(matrix_file).toarray()
dictfile=model+'2code.npy'
ccs2code=np.load(dictfile)
ccs2code=ccs2code[()]
#CCS and ATC model combinations
if model=='cat':
    model2='chem_substrs'
if model=='lvl1_':
    model2='anatoms'
if model=='lvl2_':
    model2='pharm_subgrps'
#dictionary of collapsed drug to ATC code
drugdictfile=model2+'2code.npy'
drug2code=np.load(drugdictfile)
drug2code=drug2code[()]
demo_file={demo events filename}+ case+ control + '.npy'
demo_events=np.load(demo_file)
demo_events=demo_events.tolist()
model_mat=np.zeros(shape=(matrix.shape[0],len(ccs2code.keys())+len(drug2code.keys())+len(demo_events))).astype('int8')
keys=ccs2code.keys()
for i in range(0,len(keys)):
    events=ccs2code[keys[i]]
    for e in events:
        if e in e2e.keys():
	    if e2e[e] in e2i.keys():
    		model_mat[:,i]=model_mat[:,i] | matrix[:,int(e2i[e2e[e]])]
dkeys=drug2code.keys()
for i in range(len(keys),len(keys)+len(dkeys)):
    events=drug2code[dkeys[i-len(keys)]]
    for e in events:
        if e in drug_events:
            if e in e2i.keys():
                model_mat[:,i]=model_mat[:,i] | matrix[:,int(e2i[e])]
#add demo events
for i in range(len(keys)+len(dkeys),len(keys)+len(dkeys)+len(demo_events)):
    events=demo_events
    for e in events:
	if e in e2i.keys():
	    model_mat[:,i]=matrix[:,int(e2i[e])]
model_mat=csr_matrix(model_mat)
matrix_file={test_matrix collapsed feat filename}+model+model2+case+control+'.npz'
sp.sparse.save_npz(matrix_file,model_mat)
