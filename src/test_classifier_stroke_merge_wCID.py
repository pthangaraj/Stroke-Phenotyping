#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods" 
#This script makes the test matrix for the model specified, takes arguments for case and control type
import MySQLdb
from collections import defaultdict
import sys
import csv
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import pickle
import datetime
import time
import operator
from tqdm import tqdm
from datetime import date
case=sys.argv[1]
control=sys.argv[2]
database={database name}
def credential():
        '''import login and password from credential text file'''
        reader=csv.reader(open({credentials filename}),delimiter = ",")
        for login, password in reader:
                        login=login
                        passwd=password

        return login, passwd

login,passwd=credential()

train_mrns=np.load({filename training PIDs} + case + control + '.npy')
print "first entrance to mysql", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
db = MySQLdb.connect(host={host}, user ='%s' % (login), passwd='%s' % (passwd), db=database, port={port})
c = db.cursor()
#Get test cases
SQL ='''select person_id, 0 as label from user_pt2281.rand_person_id_merge where person_id not in %s union select person_id, 1 as label from {Stroke Service Testing Cases} where person_id not in %s''' %(str(tuple(train_mrns)),str(tuple(train_mrns)))
c.execute(SQL)
results = c.fetchall()

mrns = []
mrn2label = dict()
for mrn,label in results:
    mrn = int(mrn)
    label = int(label)
    mrns.append(mrn)
    mrn2label[mrn] = label
print len(mrns), datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

cond_name={insert filename}+case+control+".npy"
cond_events=np.load(cond_name).tolist()
proc_name={insert filename}+case+control+".npy"
proc_events=np.load(proc_name).tolist()
drug_name={insert filename}+case+control+".npy"
drug_events=np.load(drug_name).tolist()
demo_name={insert filename}+case+control+".npy"
demo_events=np.load(demo_name).tolist()
rf_events=cond_events+proc_events+drug_events+demo_events
print len(rf_events)
print "getting icd9s", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
SQL = '''SELECT person_id mrn, condition_source_concept_id cid, condition_source_value icd, condition_start_date dx_date
        FROM {condition occurrence table}
	WHERE person_id in %s
        order by mrn, dx_date''' %(str(tuple(mrns)))
c.execute(SQL)
results = c.fetchall()

mrn2dx = defaultdict(lambda : defaultdict(int))
mrn_lastdate=defaultdict(lambda:date(1900,1,1))
#events = []
for mrn, cid,icd, dx_date in results:
    mrn = int(mrn)
    if cid !=0:
        icd=cid
    if str(icd) in cond_events:
        mrn2dx[mrn][str(icd)] = 1
    diff=(dx_date-mrn_lastdate[mrn]).days/365.25
    if diff>0:
        mrn_lastdate[mrn]=dx_date

print len(rf_events), len(mrn2dx), len(mrn_lastdate)
SQL = '''SELECT person_id mrn, procedure_source_concept_id pid, procedure_source_value lab, procedure_date dx_date
        FROM {procedure occurrence table}
        where person_id in %s
        order by mrn, dx_date''' %(str(tuple(mrns)))
c.execute(SQL)
results = c.fetchall()

for mrn, pid, lab, dx_date in results:
    mrn = int(mrn)
    if pid !=0:
        lab=pid
    if str(lab) in proc_events:
        mrn2dx[mrn][str(lab)] = 1
    diff=(dx_date-mrn_lastdate[mrn]).days/365.25
    if diff>0:
        mrn_lastdate[mrn]=dx_date
SQL = '''SELECT person_id mrn, drug_concept_id did, drug_era_start_date dx_date
        FROM {drug era table}
        where person_id in %s
        order by mrn, dx_date''' %(str(tuple(mrns)))
c.execute(SQL)
results = c.fetchall()

for mrn, did, dx_date in results:
    mrn = int(mrn)
    if int(did) in drug_events:
        lab=int(did)
        mrn2dx[mrn][lab] = 1
    diff=(dx_date-mrn_lastdate[mrn]).days/365.25
    if diff>0:
        mrn_lastdate[mrn]=dx_date
it=0
lastind=len(mrns)-1
for mcount in range(0,len(mrns)):
    if mcount>0 and mcount % 100000 == 0 or mcount==lastind:
        SQL = '''SELECT person_id mrn, gender_source_value gender, year_of_birth year,month_of_birth month, day_of_birth day, race_source_value race, ethnicity_source_value eth,race_concept_id race_concept_id
        FROM {person table}
        where person_id in %s
        order by mrn''' %str(tuple(mrns[it*100000:mcount]))
        c.execute(SQL)
        results = c.fetchall()
        for mrn, gender,year,month,day,race,eth,race_concept_id in results:
            mrn = int(mrn)
	    race_concept_id=int(race_concept_id)
	    if gender in demo_events:
                mrn2dx[mrn][gender] = 1
	    race_eth='NULL'	
    	    if eth=='H' or eth=='SK' or eth=='HO' or eth=='HE' or eth=='HM' or eth=='HK':
        	race_eth='Hispanic'
    	    if race=='W' and race_eth=='NULL':
        	race_eth='White'
    	    if race=='B' and race_eth=='NULL':
        	race_eth='Black'
    	    if race== 'D' and eth=='N' and race_eth=='NULL':
        	race_eth = 'Unknown'
    	    if race== 'A' or race=='AM'  and race_eth=='NULL':
        	race_eth='Asian'
    	    if race=='I' and race_eth== 'NULL':
        	race_eth='American Indian'
    	    if race=='P' or race=='PZ' and race_eth=='NULL':
        	race_eth='Pacific Islander'
    	    if race=='X' and race_eth=='NULL':
        	if race_concept_id==8516:
            	    race_eth='Black'
        	elif race_concept_id==44814653:
            	    race_eth='Unknown'
    	    if race_eth=='NULL' and race=='O':
        	race_eth='Hispanic'
    	    if race_eth=='NULL':
        	race_eth='Unknown'
	    if race_eth in demo_events:
	    	mrn2dx[mrn][race_eth]=1
            birth_date=date(int(year),int(month),int(day))
            age=(mrn_lastdate[mrn]-birth_date).days/365.25
            if age > 50:
                mrn2dx[mrn]['age']=1
            else:
                mrn2dx[mrn]['age']=0
	it+=1
event2ind=dict([(e, i) for i,e in enumerate(rf_events)])
labels=[]
print len(mrn2dx.keys()),'len xt_empi', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

matrix_test=lil_matrix((len(mrns), len(rf_events)), dtype=np.dtype(int))
for mcount, mrn in enumerate(mrns):
    if mcount % 100 == 0:
	print mcount,
    labels.append(mrn2label[mrn])
    for icd in rf_events:
        if icd in mrn2dx[mrn]:
	    matrix_test[mcount,event2ind[icd]]=mrn2dx[mrn][icd]	
filename_mrns={insert filename} + case+control+'.npy'
np.save(filename_mrns,mrns)
filename_labels={insert filename} + case + control +'.npy'
np.save(filename_labels,labels)
print "making csr_matrix", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
matrix_test=matrix_test.tocsr()
print "done with csr", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
filename_mtest={insert filename} + case + control + '.npz'
sp.sparse.save_npz(filename_mtest,matrix_test)

print '\n', len(mrn2dx)
print "done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

c.close()
db.commit()
db.close()

