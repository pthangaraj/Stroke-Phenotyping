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
def credential():
        '''import login and passwrd from credential text file'''
        reader=csv.reader(open('mycnf.csv'),delimiter = ",")
        for login, password in reader:
                        login=login
                        passwd=password

        return login, passwd

login,passwd=credential()
#def get_ancestor(descendant):
#    SQL='''select ancestor_concept_id,min_levels_of_separation from clinical_merge_v5_260318.concept_ancestor where descendant_concept_id=%s and min_levels_of_separation=1'''%str(descendant)
#    c.execute(SQL)
#    results = c.fetchall()
#    ancs=[]
#    for a,min_l in results:
#	ancs.append(a)
#    return ancs
print "first entrance to mysql", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
db = MySQLdb.connect(host='127.0.0.1', user ='%s' % (login), passwd='%s' % (passwd), db='clinical_gm', port=3307)
c = db.cursor()
cid2icd=dict()
cond_file='cond_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy'
cond_events=np.load(cond_file)
cond_events=cond_events.tolist()

SQL='''select condition_source_value icd, condition_source_concept_id cid from clinical_merge_v5_260318.condition_occurrence where condition_source_concept_id in %s''' %str(tuple(cond_events))
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
proc_file='proc_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy'
proc_events=np.load(proc_file)
proc_events=proc_events.tolist()
SQL='''select procedure_source_value icd, procedure_source_concept_id cid from clinical_merge_v5_260318.procedure_occurrence where procedure_source_concept_id in %s''' %str(tuple(proc_events))
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
e2e_file='events2uniformevents_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy'
np.save(e2e_file,e2e)
drug_file='drug_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy'
drug_events=np.load(drug_file)
drug_events=drug_events.tolist()
e2i_file='events2colswCID_1to1_cond_dera_sou_merge_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy'
e2i=np.load(e2i_file)
e2i=e2i[()]
matrix_file='matrix_trainwCID_1to1_cond_dera_sou_merge_raceeth_mergeset_noctrlicd_new_fixcv' + case + control + '.npz'
matrix=sp.sparse.load_npz(matrix_file).toarray()
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
demo_file='demo_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy'
demo_events=np.load(demo_file)
demo_events=demo_events.tolist()
#e2e_file='events2uniformeventswCID_1to1_cond_dera'+case+control+'.npy'
#np.save(e2e_file,e2e)
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
#y='train_labelswCID_1to1_cond_dera' + case + control + '.npy'
#labels=np.load(y)
examples=csr_matrix(model_mat)
mat_file='matrix_trainwCID_1to1_cond_dera_sou_merge_raceeth_mergeset_noctrlicd_new_fixcv'+model+model2+case+control+'.npz'
sp.sparse.save_npz(mat_file,examples)
print "lr start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
