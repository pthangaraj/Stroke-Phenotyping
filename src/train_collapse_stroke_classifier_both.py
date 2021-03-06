#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods" 
#This script makes the training matrix with collapsed features
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
print "new loop start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
def credential():
        '''import login and passwrd from credential text file'''
        reader=csv.reader(open({credentials filename}),delimiter = ",")
        for login, password in reader:
                        login=login
                        passwd=password

        return login, passwd

login,passwd=credential()
print "first entrance to mysql", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
db = MySQLdb.connect(host={host}, user ='%s' % (login), passwd='%s' % (passwd), db={database}, port={poat})
c = db.cursor()
cid2icd=dict()
cond_file={condition events filename}+case+control+'.npy'
cond_events=np.load(cond_file)
cond_events=cond_events.tolist()
#gather ICD9 or ICD10 codes of conditions
SQL='''select condition_source_value icd, condition_source_concept_id cid from {condition occurrence table} where condition_source_concept_id in %s''' %str(tuple(cond_events))
c.execute(SQL)
results = c.fetchall()

for icd,cid in results:
    cid2icd[cid]=icd
#snomed concept id, 
for cond in cond_events:
    if cond in cid2icd.keys():
        cond=cid2icd[cond]
    ocond=cond
    if cond[0:3]=='I9:':
	print cond
	cond=cond.split(':')[1]
    elif cond[0:4]=='I10:':
	cond=cond.split(':')[1]
    e2e[cond]=ocond
proc_file={procedure events filename}+case+control+'.npy'
proc_events=np.load(proc_file)
proc_events=proc_events.tolist()
#Gather ICD9 or ICD10 codes of procedures
SQL='''select procedure_source_value icd, procedure_source_concept_id cid from {procedure occurrence table} where procedure_source_concept_id in %s''' %str(tuple(proc_events))
c.execute(SQL)
results = c.fetchall()
for icd,cid in results:
    cid2icd[cid]=icd
for proc in proc_events:
    if proc in cid2icd.keys():
	proc=cid2icd[proc]
    oproc=proc
    if proc[0:3]=='I9:':
        proc=proc.split(':')[1]
    elif proc[0:4]=='I10:':
        proc=proc.split(':')[1]
    elif proc[0:3]=='C4:':
	proc=proc.split(':')[1]
    e2e[proc]=oproc
e2e_file={events2uniformevents filename}+case+control+'.npy'
np.save(e2e_file,e2e)
drug_file={drug era events filename}+case+control+'.npy'
drug_events=np.load(drug_file)
drug_events=drug_events.tolist()
e2i_file={events2cols filename}+case+control+'.npy'
e2i=np.load(e2i_file)
e2i=e2i[()]
matrix_file={training_set matrix filename} + case + control + '.npz'
matrix=sp.sparse.load_npz(matrix_file).toarray()
#load dictionary of feature collapsing models based on CCS+ATC combination
dictfile=model+'2code.npy'
ccs2code=np.load(dictfile)
ccs2code=ccs2code[()]
if model=='cat':
    model2='chem_substrs'
if model=='lvl1_':
    model2='anatoms'
if model=='lvl2_':
    model2='pharm_subgrps'
drugdictfile=model2+'2code.npy'
drug2code=np.load(drugdictfile)
drug2code=drug2code[()]
demo_file={demographics filename}+case+control+'.npy'
demo_events=np.load(demo_file)
demo_events=demo_events.tolist()
#matrix of collapsed features
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
	
C_val = 1
examples=csr_matrix(model_mat)
mat_file={insert matrix filename}+model+model2+case+control+'.npz'
sp.sparse.save_npz(mat_file,examples)
print "end", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
