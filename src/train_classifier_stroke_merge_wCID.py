#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods"  
#adapted from Tal Lorberbaum's ipython notebook code
import MySQLdb
from collections import defaultdict
import sys
import csv
import numpy as np
import random
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
#import keras
#from keras.models import Sequential
#from keras.layers import Dense

case=sys.argv[1]
control=sys.argv[2]
#lvl=sys.argv[3]
def credential():
        '''import login and passwrd from credential text file'''
        reader=csv.reader(open('mycnf.csv'),delimiter = ",")
        for login, password in reader:
                        login=login
                        passwd=password

        return login, passwd

login,passwd=credential()
print "first entrance to mysql", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
db = MySQLdb.connect(host='127.0.0.1', user ='%s' % (login), passwd='%s' % (passwd), db='clinical_gm', port=3307)
c = db.cursor()
#first set the 
if case=='G':
    SQLsCa = '''from {Stroke Service Training Table}'''
elif case == 'T':
    SQLsCa = '''from {Tirschwell criteria Table}  where person_id not in (select person_id from {Stroke Service Testing Table}) order by rand() limit 8000'''
elif case == 'C':
    SQLsCa = '''from {CCS cerebrovascular disease Table} where person_id not in (select person_id from {Stroke Service Testing Table}) order by rand() limit 8000'''


SQL='''select person_id, 1 as label %s''' %(SQLsCa)
c.execute(SQL)
results=c.fetchall()

mrn2label = dict()
pos_mrns=list()
for mrn, label in results:
    mrn = int(mrn)
    label = int(label)
    pos_mrns.append(mrn)
    mrn2label[mrn] = label
#Stroke mimetics
if control=='N':
    SQLsCo = '''select c.person_id, 0 as label from {Stroke mimetic Controls Table} join {Patient mapppings table} c where mrn=existing_patient_id and c.source_id_type='Eagle_MRN' and c.person_id not in (select person_id from {Stroke Service Testing Table} union select person_id from {Tirschwell criteria Table})'''
#ANYONE not in case
if control == 'R':
   SQLsCo = '''select c.person_id, 0 as label from {condition occurrence table} c join {person table} p where c.person_id=p.person_id and c.person_id not in %s and c.person_id not in (select person_id from {Stroke Service Testing Table})''' % str(tuple(pos_mrns))
#people with no CCS at all
elif control == 'C':
    SQLsCo =  '''select person_id, 0 as label from {No CCS cerebrovascular disease Table} where person_id not in %s and person_id not in (select person_id from {Stroke Service Testing Table})'''% str(tuple(pos_mrns))
#people with CCS but no stroke at all
elif control == 'CI':
    SQLsCo =  '''select person_id, 0 as label from {CCS cerebrovascular disease without Tirschwell criteria Table} where person_id not in %s and person_id not in (select person_id from {Stroke Service Testing Table})'''% str(tuple(pos_mrns))
#people with no stroke at all (so GS+tirschwell)
elif control == 'I':
    SQLsCo =  '''select person_id, 0 as label from {No Tirschwell criteria Table} where person_id not in %s and person_id not in(select person_id from {Stroke Service Testing Table})''' % str(tuple(pos_mrns))
SQL='''%s''' % (SQLsCo)
c.execute(SQL)
results=c.fetchall()

#Making 1:1: case:control ratio
neg_mrns=list()
for mrn, label in results:
    mrn = int(mrn)
    label = int(label)
    neg_mrns.append(mrn)
    mrn2label[mrn] = label
if len(pos_mrns)<len(neg_mrns):
        neg_cont=random.sample(neg_mrns,len(pos_mrns))
	pos_cont=pos_mrns
elif len(pos_mrns)>len(neg_mrns):
	pos_cont=random.sample(pos_mrns,len(neg_mrns))
	neg_cont=neg_mrns
else:
	neg_cont=neg_mrns
	pos_cont=pos_mrns
print len(neg_cont), len(pos_cont),  len(mrn2label)

mrns=neg_cont+pos_cont
#save training person_ids
filename_mrns={insert filename}+case+control+".npy"
np.save(filename_mrns,mrns)
print "get cohorts", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
# Obtain time-ordered Dx codes and dates for each patient, excluding all codes used
# to generate cohorts
#Gather conditions
SQL = '''SELECT person_id, condition_source_concept_id cid, condition_source_value icd, condition_start_date dx_date
        FROM {condition occurrence table}
        WHERE condition_source_value not in (select icd from {ICDs to generate cases and controls Table})
        and person_id in %s
        order by person_id, dx_date''' %str(tuple(mrns))
c.execute(SQL)
results = c.fetchall()

mrn2dx = defaultdict(dict)
mrn_lastdate=defaultdict(lambda:date(1900,1,1))
events = []
cond_events=[]
for mrn, cid, icd, dx_date in results:
    mrn = int(mrn)
    if cid !=0:
	icd=cid
    if icd not in events:
        events.append(icd)
	cond_events.append(icd)
    mrn2dx[mrn][icd] = 1
    diff=(dx_date-mrn_lastdate[mrn]).days/365.25
    if diff>0:
	mrn_lastdate[mrn]=dx_date
   
events = sorted(events)
print len(events), len(mrn2dx), len(mrn_lastdate)
#Gather procedures
SQL = '''SELECT person_id, procedure_source_concept_id pid, procedure_source_value lab, procedure_date dx_date
        FROM {procedure occurrence table}
        where person_id in %s
        order by person_id, dx_date''' %str(tuple(mrns))
c.execute(SQL)
results = c.fetchall()
proc_events=[]
for mrn, pid, lab, dx_date in results:
    mrn = int(mrn)
    if pid !=0:
        lab=pid
    if lab not in events:
	events.append(lab)
	proc_events.append(lab)
    mrn2dx[mrn][lab] = 1
    diff=(dx_date-mrn_lastdate[mrn]).days/365.25
    if diff>0:
        mrn_lastdate[mrn]=dx_date
#gather medication orders
SQL = '''SELECT person_id, drug_concept_id did, drug_era_start_date dx_date
        FROM {drug era table}
        where person_id in %s
        order by person_id, dx_date''' %str(tuple(mrns))
c.execute(SQL)
results = c.fetchall()
drug_events=[]
for mrn, did, dx_date in results:
    mrn = int(mrn)
    lab=did
    if lab not in events:
	events.append(lab)
	drug_events.append(lab)
    mrn2dx[mrn][lab] = 1
    diff=(dx_date-mrn_lastdate[mrn]).days/365.25
    if diff>0:
        mrn_lastdate[mrn]=dx_date
#Gather gender, ethnicity, race, age
SQL = '''SELECT person_id, gender_source_value gender, year_of_birth year,month_of_birth month, day_of_birth day, race_source_value race, ethnicity_source_value eth,race_concept_id race_concept_id
        FROM {person table}
        where person_id in %s
        order by person_id''' %str(tuple(mrns))
c.execute(SQL)
results = c.fetchall()
demo_events=[]
for mrn, gender,year,month,day,race,eth,race_concept_id in results:
    mrn = int(mrn)
    race_concept_id=int(race_concept_id)
    if gender not in events: 
        events.append(gender)
	demo_events.append(gender)
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
    if race_eth not in events:
        events.append(race_eth)
        demo_events.append(race_eth)
    mrn2dx[mrn][gender] = 1
    mrn2dx[mrn][race_eth]=1
    try:
	birth_date=date(int(year),int(month),int(day))
    except ValueError:
	birth_date=date(int(year),int(month),1)
    age=(mrn_lastdate[mrn]-birth_date).days/365.25
    if age > 0:
	mrn2dx[mrn]['age']=age
    else:
	mrn2dx[mrn]['age']=-1
demo_events.append('age')
labels = []
rows = []
e2i={}
mrn2label_final=dict()
for mcount, mrn in enumerate(mrn2dx):
    if mcount % 100 == 0:
        print mcount,
    mrn2label_final[mrn]=mrn2label[mrn]   
    labels.append(mrn2label[mrn])
    row = []
    col=0
    for icd in events:
	if mcount==0:
	    e2i[icd]=col
	    col=col+1
        if icd in mrn2dx[mrn]:
            row.append(1)
        else:
            row.append(0)
    if 'age' in mrn2dx[mrn].keys():
        if mrn2dx[mrn]['age']>=50:
	    row.append(1)
        else:
	    row.append(0)
    else:
	row.append(0)
    if mcount==0:
        e2i['age']=col
        
    #row.append(mrn2dx[mrn]['age'])
    rows.append(row)
print '\n', len(mrn2dx),len(rows)
matrix_arr = csr_matrix(rows)
filename_mtrain={insert filename} + case + control + '.npz'
sp.sparse.save_npz(filename_mtrain, matrix_arr)
filename_labels={insert filename} + case + control + '.npy'
np.save(filename_labels,labels)
filename_mrn2label={insert filename} + case + control + '.npy'
np.save(filename_mrn2label, mrn2label_final)
filename_e2i={insert filename}+case+control+'.npy'
np.save(filename_e2i,e2i)
filename_all={insert filename} + case + control + '.npy'
np.save(filename_all,tuple(cond_events))
filename_all={insert filename} + case + control + '.npy'
np.save(filename_all,tuple(proc_events))
filename_all={insert filename} + case + control + '.npy'
np.save(filename_all,tuple(drug_events))
filename_all={insert filename} + case + control + '.npy'
np.save(filename_all,tuple(demo_events))
c.close()
db.commit()
db.close()

