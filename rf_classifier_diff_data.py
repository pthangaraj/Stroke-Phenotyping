#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods
#This script is to build a Generalized linear model to determine feature category contribution to models
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
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from scipy.sparse import csr_matrix,hstack
import pickle
import time
import datetime
from datetime import date

case=sys.argv[1]
control=sys.argv[2]
coefs=np.zeros((5,4,2))
pvals=np.zeros((5,4,2))
roc_metrics=np.zeros((5,5))

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
train_mrns=np.load({training_set_pids_filename} + case + control + '.npy')
mrn2label=np.load({training_mrn2labels_filename} + case + control + '.npy')
mrn2label=mrn2label[()]
train_mrns=list(set(train_mrns)&set(mrn2label.keys()))
#build tw model
SQL = '''SELECT person_id, condition_source_concept_id cid, condition_source_value icd, condition_start_date dx_date
        FROM {condition_occurrence_table}
        WHERE condition_source_value in (select icd from {tirsch_criteria_table})
        and person_id in %s
        order by person_id, dx_date''' %str(tuple(train_mrns))
c.execute(SQL)
results = c.fetchall() 
mrn2dx = defaultdict(dict)
mrn2date_dx = defaultdict(lambda: defaultdict(dict))
tirsch_events=[]
for mrn, cid, icd, dx_date in results:
    mrn = int(mrn)
    if cid !=0:
        icd=int(cid)
    if icd not in tirsch_events:
        tirsch_events.append(icd)
    mrn2dx[mrn][icd] = 1
labels = []
rows = []
e2i={}
for mcount, mrn in enumerate(train_mrns):
    if mcount % 100 == 0:
        print mcount,
    labels.append(mrn2label[mrn])
    row = []
    col=0
    for icd in tirsch_events:
        if mcount==0:
            e2i[icd]=col
            col=col+1
        if icd in mrn2dx[mrn]:
            row.append(1)
        else:
            row.append(0)
    rows.append(row)
print '\n', len(mrn2dx),len(rows)
X_tr_tw = csr_matrix(rows)

test_mrns=np.load( + case+control+'.npy')
SQL = '''SELECT person_id, condition_source_concept_id cid, condition_source_value icd, condition_start_date dx_date
        FROM {condition_occurrence_table}
        WHERE condition_source_value in (select icd from {tirsch_criteria_table})
        and person_id in %s
        order by person_id, dx_date''' %str(tuple(test_mrns))
c.execute(SQL)
results = c.fetchall()
mrn2dx = defaultdict(dict)
mrn2date_dx = defaultdict(lambda: defaultdict(dict))
filename_all={tirsch_events_filename} + case + control + '.npy'
np.save(filename_all,tirsch_events)
#print tirsch_events
for mrn, cid,icd, dx_date in results:
    mrn = int(mrn)
    if cid !=0:
        icd=int(cid)
    if icd in tirsch_events:
        mrn2dx[mrn][icd] = 1
    elif icd in tirsch_events:
	mrn2dx[mrn][icd] = 1
#labels = []
rows = []
e2i={}
for mcount, mrn in enumerate(test_mrns):
    if mcount % 100 == 0:
        print mcount,
    #labels.append(mrn2label[mrn])
    row = []
    col=0
    for icd in tirsch_events:
        if mcount==0:
            e2i[icd]=col
            col=col+1
        if icd in mrn2dx[mrn]:
            row.append(1)
        else:
            row.append(0)
    rows.append(row)
print '\n', len(mrn2dx),len(rows)
X_te_tw = csr_matrix(rows)
#build model on tw
print "done getting tirsch_events", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
test_labels_filename={test_labels_filename}+case+control+".npy"
test_labels=np.load(test_labels_filename)
y=labels
C_val=.1
lr_tw=linear_model.LogisticRegression(penalty = 'l1', C = C_val)
ab_tw= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=1000)
rf_tw= ensemble.RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, oob_score=True)
gb_tw=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=1000, max_features="sqrt", max_depth=10)
en_tw=linear_model.SGDClassifier(penalty='elasticnet',l1_ratio=.01,alpha=.01,loss='log')
lr_tw.fit(X_tr_tw,y)
lr_label_tw=lr_tw.predict_proba(X_te_tw)
roc_metrics[0,0]=metrics.roc_auc_score(test_labels,lr_label_tw[:,1])
#print "lr auc-roc:",metrics.roc_auc_score(test_labels,lr_label_tw[:,1])
rf_tw.fit(X_tr_tw,y)
rf_label_tw=rf_tw.predict_proba(X_te_tw)
roc_metrics[1,0]=metrics.roc_auc_score(test_labels,rf_label_tw[:,1])
#print "rf auc-roc:",metrics.roc_auc_score(test_labels,rf_label_tw[:,1])
ab_tw.fit(X_tr_tw,y)
ab_label_tw=ab_tw.predict_proba(X_te_tw)
roc_metrics[2,0]=metrics.roc_auc_score(test_labels,ab_label_tw[:,1])
#print "ab auc-roc:",metrics.roc_auc_score(test_labels,ab_label_tw[:,1])
gb_tw.fit(X_tr_tw,y)
gb_label_tw=gb_tw.predict_proba(X_te_tw)
#print "gb auc-roc:",metrics.roc_auc_score(test_labels,gb_label_tw[:,1])
roc_metrics[3,0]=metrics.roc_auc_score(test_labels,gb_label_tw[:,1])
en_tw.fit(X_tr_tw,y)
en_label_tw=en_tw.predict_proba(X_te_tw)
#print "en auc-roc:",metrics.roc_auc_score(test_labels,en_label_tw[:,1])
roc_metrics[4,0]=metrics.roc_auc_score(test_labels,en_label_tw[:,1])
filename_mtrain={training_set_matrix_filename} + case + control + '.npz'
X_tr=sp.sparse.load_npz(filename_mtrain)

filename_mtest={testing_set_matrix_filename} + case + control + '.npz'
X_te=sp.sparse.load_npz(filename_mtest)
filename_e2i={events2columns_filename}+case+control+'.npy'
e2i=np.load(filename_e2i)
e2i=e2i[()]
filename_labels={training_set_labels_filename} + case + control + '.npy'
y=np.load(filename_labels)
#load conditions
filename_all={condition_events_filename} + case + control + '.npy'
cond_events=np.load(filename_all)
cond_args=[]
for event in cond_events:
    if event not in e2i.keys():
        cond_args.append(int(e2i[int(event)]))
    else:
	cond_args.append(int(e2i[event]))
X_tr_co=X_tr[:,cond_args]
X_te_co=X_te[:,cond_args]
#build model on conditions
print "done getting cond_events", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d# %H:%M:%S')
lr_co=linear_model.LogisticRegression(penalty = 'l1', C = C_val)
ab_co= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=1000)
rf_co= ensemble.RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, oob_score=True)
gb_co=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=1000, max_features="sqrt", max_depth=10)
en_co=linear_model.SGDClassifier(penalty='elasticnet',l1_ratio=.01,alpha=.01,loss='log')

lr_co.fit(X_tr_co,y)
lr_label_co=lr_co.predict_proba(X_te_co)
np.save("lr_label_co_probs"+case+control+".npy",lr_label_co)
#print "lr auc-roc:",metrics.roc_auc_score(test_labels,lr_label_co[:,1])
roc_metrics[0,1]=metrics.roc_auc_score(test_labels,lr_label_tw[:,1])
rf_co.fit(X_tr_co,y)
rf_label_co=rf_co.predict_proba(X_te_co)
np.save("rf_label_co_probs"+case+control+".npy",rf_label_co)
#print "rf auc-roc:",metrics.roc_auc_score(test_labels,rf_label_co[:,1])
roc_metrics[1,1]=metrics.roc_auc_score(test_labels,rf_label_tw[:,1])
ab_co.fit(X_tr_co,y)
ab_label_co=ab_co.predict_proba(X_te_co)
np.save("ab_label_co_probs"+case+control+".npy",ab_label_co)
#print "ab auc-roc:",metrics.roc_auc_score(test_labels,ab_label_co[:,1])
roc_metrics[2,1]=metrics.roc_auc_score(test_labels,ab_label_tw[:,1])
gb_co.fit(X_tr_co,y)
gb_label_co=gb_co.predict_proba(X_te_co)
np.save("gb_label_co_probs"+case+control+".npy",gb_label_co)
#print "gb auc-roc:",metrics.roc_auc_score(test_labels,gb_label_co[:,1])
roc_metrics[3,1]=metrics.roc_auc_score(test_labels,gb_label_tw[:,1])
#print "done cond_model", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
en_co.fit(X_tr_co,y)
en_label_co=en_co.predict_proba(X_te_co)
np.save("en_label_co_probs"+case+control+".npy",en_label_co)
#print "en auc-roc:",metrics.roc_auc_score(test_labels,en_label_co[:,1])
roc_metrics[4,1]=metrics.roc_auc_score(test_labels,en_label_tw[:,1])
print "done cond_model", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
filename_all={procedure_events_filename} + case + control + '.npy'
proc_events=np.load(filename_all)
proc_args=[]
for event in proc_events:
    if event not in e2i.keys():
        proc_args.append(int(e2i[int(event)]))
    else:
        proc_args.append(int(e2i[event]))
X_tr_pr=X_tr[:,proc_args]
X_te_pr=X_te[:,proc_args]
#build model on procedures
lr_pr=linear_model.LogisticRegression(penalty = 'l1', C = C_val)
ab_pr= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=1000)
rf_pr= ensemble.RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, oob_score=True)
gb_pr=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=1000, max_features="sqrt", max_depth=10)
en_pr=linear_model.SGDClassifier(penalty='elasticnet',l1_ratio=.01,alpha=.01,loss='log')

lr_pr.fit(X_tr_pr,y)
lr_label_pr=lr_pr.predict_proba(X_te_pr)
np.save("lr_label_pr_probs"+case+control+".npy",lr_label_pr)
#print "lr auc-roc:",metrics.roc_auc_score(test_labels,lr_label_pr[:,1])
roc_metrics[0,2]=metrics.roc_auc_score(test_labels,lr_label_tw[:,1])
rf_pr.fit(X_tr_pr,y)
rf_label_pr=rf_pr.predict_proba(X_te_pr)
np.save("rf_label_pr_probs"+case+control+".npy",rf_label_pr)
#print "rf auc-roc:",metrics.roc_auc_score(test_labels,rf_label_pr[:,1])
roc_metrics[1,2]=metrics.roc_auc_score(test_labels,rf_label_tw[:,1])
ab_pr.fit(X_tr_pr,y)
ab_label_pr=ab_pr.predict_proba(X_te_pr)
np.save("ab_label_pr_probs"+case+control+".npy",ab_label_pr)
#print "ab auc-roc:",metrics.roc_auc_score(test_labels,ab_label_pr[:,1])
roc_metrics[1,2]=metrics.roc_auc_score(test_labels,ab_label_tw[:,1])
gb_pr.fit(X_tr_pr,y)
gb_label_pr=gb_pr.predict_proba(X_te_pr)
np.save("gb_label_pr_probs"+case+control+".npy",gb_label_pr)
#print "gb auc-roc:",metrics.roc_auc_score(test_labels,gb_label_pr[:,1])
roc_metrics[3,2]=metrics.roc_auc_score(test_labels,gb_label_tw[:,1])
en_pr.fit(X_tr_pr,y)
en_label_pr=en_pr.predict_proba(X_te_pr)
np.save("en_label_pr_probs"+case+control+".npy",en_label_pr)
#print "en auc-roc:",metrics.roc_auc_score(test_labels,en_label_pr[:,1])
roc_metrics[4,2]=metrics.roc_auc_score(test_labels,en_label_tw[:,1])
print "done proc_model", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#load demographics
filename_all={demo_events_filename} + case + control + '.npy'
demo_events=np.load(filename_all)
demo_args=[]
for event in demo_events:
    if event not in e2i.keys():
        demo_args.append(int(e2i[int(event)]))
    else:
        demo_args.append(int(e2i[event]))
X_tr_de=X_tr[:,demo_args]
X_te_de=X_te[:,demo_args]
#build model on demographics
lr_de=linear_model.LogisticRegression(penalty = 'l1', C = C_val)
ab_de= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=200)
rf_de= ensemble.RandomForestClassifier(n_estimators=200, max_features="sqrt", max_depth=10, oob_score=True)
gb_de=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=200, max_features="sqrt", max_depth=10)
en_de=linear_model.SGDClassifier(penalty='elasticnet',l1_ratio=.01,alpha=.01,loss='log')

lr_de.fit(X_tr_de,y)
lr_label_de=lr_de.predict_proba(X_te_de)
np.save("lr_label_de_probs"+case+control+".npy",lr_label_de)
print "lr auc-roc:",metrics.roc_auc_score(test_labels,lr_label_de[:,1])
rf_de.fit(X_tr_de,y)
rf_label_de=rf_de.predict_proba(X_te_de)
np.save("rf_label_de_probs"+case+control+".npy",rf_label_de)
print "rf auc-roc:",metrics.roc_auc_score(test_labels,rf_label_de[:,1])
ab_de.fit(X_tr_de,y)
ab_label_de=ab_de.predict_proba(X_te_de)
np.save("ab_label_de_probs"+case+control+".npy",ab_label_de)
print "ab auc-roc:",metrics.roc_auc_score(test_labels,ab_label_de[:,1])
gb_de.fit(X_tr_de,y)
gb_label_de=gb_de.predict_proba(X_te_de)
np.save("gb_label_de_probs"+case+control+".npy",gb_label_de)
print "gb auc-roc:",metrics.roc_auc_score(test_labels,gb_label_de[:,1])
en_de.fit(X_tr_de,y)
en_label_de=en_de.predict_proba(X_te_de)
np.save("en_label_de_probs"+case+control+".npy",en_label_de)
print "en auc-roc:",metrics.roc_auc_score(test_labels,en_label_de[:,1])
print "done demo_model", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#load drugs
filename_all={drug_events_filename} + case + control + '.npy'
drug_events=np.load(filename_all)
drug_args=[]
for event in drug_events:
    if event not in e2i.keys():
        drug_args.append(int(e2i[int(event)]))
    else:
        drug_args.append(int(e2i[event]))
X_tr_dr=X_tr[:,drug_args]
X_te_dr=X_te[:,drug_args]
#build model on drugs
#print "rf auc-roc:",metrics.roc_auc_score(test_labels,rf_label_dr[:,1])
lr_dr=linear_model.LogisticRegression(penalty = 'l1', C = C_val)
ab_dr= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=200)
rf_dr= ensemble.RandomForestClassifier(n_estimators=200, max_features="sqrt", max_depth=10, oob_score=True)
gb_dr=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=200, max_features="sqrt", max_depth=10)
en_dr=linear_model.SGDClassifier(penalty='elasticnet',l1_ratio=.01,alpha=.01,loss='log')
#parallelize this
lr_dr.fit(X_tr_dr,y)
lr_label_dr=lr_dr.predict_proba(X_te_dr)
np.save("lr_label_dr_probs"+case+control+".npy",lr_label_dr)
print "lr auc-roc:",metrics.roc_auc_score(test_labels,lr_label_dr[:,1])
rf_dr.fit(X_tr_dr,y)
rf_label_dr=rf_dr.predict_proba(X_te_dr)
np.save("rf_label_dr_probs"+case+control+".npy",rf_label_dr)
print "rf auc-roc:",metrics.roc_auc_score(test_labels,rf_label_dr[:,1])
ab_dr.fit(X_tr_dr,y)
ab_label_dr=ab_dr.predict_proba(X_te_dr)
np.save("ab_label_dr_probs"+case+control+".npy",ab_label_dr)
print "ab auc-roc:",metrics.roc_auc_score(test_labels,ab_label_dr[:,1])
gb_dr.fit(X_tr_dr,y)
gb_label_dr=gb_dr.predict_proba(X_te_dr)
np.save("gb_label_dr_probs"+case+control+".npy",gb_label_dr)
print "gb auc-roc:",metrics.roc_auc_score(test_labels,gb_label_dr[:,1])
en_dr.fit(X_tr_dr,y)
en_label_dr=en_dr.predict_proba(X_te_dr)
np.save("en_label_dr_probs"+case+control+".npy",en_label_dr)
print "en auc-roc:",metrics.roc_auc_score(test_labels,en_label_dr[:,1])
#print lr_label_dr, rf_label_dr, ab_label_dr, gb_label_dr
print "done drug_model", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

import statsmodels.api as sm
import statsmodels.tools.tools
##Normalized with Tirschwell
prob_matrix_lr=np.vstack(((lr_label_tw[:,1]-np.mean(lr_label_tw[:,1]))/np.std(lr_label_tw[:,1]),(lr_label_co[:,1]-np.mean(lr_label_co[:,1]))/np.std(lr_label_co[:,1]),(lr_label_pr[:,1]-np.mean(lr_label_pr[:,1]))/np.std(lr_label_pr[:,1]),(lr_label_de[:,1]-np.mean(lr_label_de[:,1]))/np.std(lr_label_de[:,1]),(lr_label_dr[:,1]-np.mean(lr_label_dr[:,1]))/np.std(lr_label_dr[:,1]))).T
prob_matrix_rf=np.vstack(((rf_label_tw[:,1]-np.mean(rf_label_tw[:,1]))/np.std(rf_label_tw[:,1]),(rf_label_co[:,1]-np.mean(rf_label_co[:,1]))/np.std(rf_label_co[:,1]),(rf_label_pr[:,1]-np.mean(rf_label_pr[:,1]))/np.std(rf_label_pr[:,1]),(rf_label_de[:,1]-np.mean(rf_label_de[:,1]))/np.std(rf_label_de[:,1]),(rf_label_dr[:,1]-np.mean(rf_label_dr[:,1]))/np.std(rf_label_dr[:,1]))).T
prob_matrix_ab=np.vstack(((ab_label_tw[:,1]-np.mean(ab_label_tw[:,1]))/np.std(ab_label_tw[:,1]),(ab_label_co[:,1]-np.mean(ab_label_co[:,1]))/np.std(ab_label_co[:,1]),(ab_label_pr[:,1]-np.mean(ab_label_pr[:,1]))/np.std(ab_label_pr[:,1]),(ab_label_de[:,1]-np.mean(ab_label_de[:,1]))/np.std(ab_label_de[:,1]),(ab_label_dr[:,1]-np.mean(ab_label_dr[:,1]))/np.std(ab_label_dr[:,1]))).T
prob_matrix_gb=np.vstack(((gb_label_tw[:,1]-np.mean(gb_label_tw[:,1]))/np.std(gb_label_tw[:,1]),(gb_label_co[:,1]-np.mean(gb_label_co[:,1]))/np.std(gb_label_co[:,1]),(gb_label_pr[:,1]-np.mean(gb_label_pr[:,1]))/np.std(gb_label_pr[:,1]),(gb_label_de[:,1]-np.mean(gb_label_de[:,1]))/np.std(gb_label_de[:,1]),(gb_label_dr[:,1]-np.mean(gb_label_dr[:,1]))/np.std(gb_label_dr[:,1]))).T
prob_matrix_en=np.vstack(((en_label_tw[:,1]-np.mean(en_label_tw[:,1]))/np.std(en_label_tw[:,1]),(en_label_co[:,1]-np.mean(en_label_co[:,1]))/np.std(en_label_co[:,1]),(en_label_pr[:,1]-np.mean(en_label_pr[:,1]))/np.std(en_label_pr[:,1]),(en_label_de[:,1]-np.mean(en_label_de[:,1]))/np.std(en_label_de[:,1]),(en_label_dr[:,1]-np.mean(en_label_dr[:,1]))/np.std(en_label_dr[:,1]))).T
print np.mean(prob_matrix_lr,axis=0)
print np.std(prob_matrix_lr,axis=0)
print np.mean(prob_matrix_rf,axis=0)
print np.std(prob_matrix_rf,axis=0)
print np.mean(prob_matrix_ab,axis=0)
print np.std(prob_matrix_ab,axis=0)
print np.mean(prob_matrix_gb,axis=0)
print np.std(prob_matrix_gb,axis=0)
print np.mean(prob_matrix_en,axis=0)
print np.std(prob_matrix_en,axis=0)
print "begin GLM", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
prob_matrix_lr=statsmodels.tools.tools.add_constant(prob_matrix_lr, prepend=True,has_constant='add')
print prob_matrix_lr.shape
prob_matrix_rf=statsmodels.tools.tools.add_constant(prob_matrix_rf, prepend=True,has_constant='add')
#prob_matrix_rf=sm.add_constant(prob_matrix_rf,has_constant='skip')
print prob_matrix_rf.shape
prob_matrix_ab=statsmodels.tools.tools.add_constant(prob_matrix_ab, prepend=True,has_constant='add')
print prob_matrix_ab.shape
prob_matrix_gb=statsmodels.tools.tools.add_constant(prob_matrix_gb, prepend=True,has_constant='add')
print prob_matrix_gb.shape
prob_matrix_en=statsmodels.tools.tools.add_constant(prob_matrix_en, prepend=True,has_constant='add')
print prob_matrix_en.shape
bin_model_lr=sm.GLM(test_labels,prob_matrix_lr,family=sm.families.Binomial())
results_lr=bin_model_lr.fit()
print("lr_all",results_lr.summary())
bin_model_rf=sm.GLM(test_labels,prob_matrix_rf,family=sm.families.Binomial())
results_rf=bin_model_rf.fit()
print("rf_all",results_rf.summary())
bin_model_ab=sm.GLM(test_labels,prob_matrix_ab,family=sm.families.Binomial())
results_ab=bin_model_ab.fit()
print("ab_all",results_ab.summary())
bin_model_gb=sm.GLM(test_labels,prob_matrix_gb,family=sm.families.Binomial())
results_gb=bin_model_gb.fit()
print("gb_all",results_gb.summary())
bin_model_en=sm.GLM(test_labels,prob_matrix_en,family=sm.families.Binomial())
results_en=bin_model_en.fit()
print("en_all",results_en.summary())
print "done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
##tw vs cond
#prob_matrix_lr_co=np.vstack(((lr_label_tw[:,1]-np.mean(lr_label_tw[:,1]))/np.std(lr_label_tw[:,1]),(lr_label_co[:,1]-np.mean(lr_label_co[:,1]))/np.std(lr_label_co[:,1]))).T
#prob_matrix_rf_co=np.vstack(((rf_label_tw[:,1]-np.mean(rf_label_tw[:,1]))/np.std(rf_label_tw[:,1]),(rf_label_co[:,1]-np.mean(rf_label_co[:,1]))/np.std(rf_label_co[:,1]))).T
#prob_matrix_ab_co=np.vstack(((ab_label_tw[:,1]-np.mean(ab_label_tw[:,1]))/np.std(ab_label_tw[:,1]),(ab_label_co[:,1]-np.mean(ab_label_co[:,1]))/np.std(ab_label_co[:,1]))).T
#prob_matrix_gb_co=np.vstack(((gb_label_tw[:,1]-np.mean(gb_label_tw[:,1]))/np.std(gb_label_tw[:,1]),(gb_label_co[:,1]-np.mean(gb_label_co[:,1]))/np.std(gb_label_co[:,1]))).T
#prob_matrix_en_co=np.vstack(((en_label_tw[:,1]-np.mean(en_label_tw[:,1]))/np.std(en_label_tw[:,1]),(en_label_co[:,1]-np.mean(en_label_co[:,1]))/np.std(en_label_co[:,1]))).T
#
##tw vs proc
#prob_matrix_lr_pr=np.vstack(((lr_label_tw[:,1]-np.mean(lr_label_tw[:,1]))/np.std(lr_label_tw[:,1]),(lr_label_pr[:,1]-np.mean(lr_label_pr[:,1]))/np.std(lr_label_pr[:,1]))).T
#prob_matrix_rf_pr=np.vstack(((rf_label_tw[:,1]-np.mean(rf_label_tw[:,1]))/np.std(rf_label_tw[:,1]),(rf_label_pr[:,1]-np.mean(rf_label_pr[:,1]))/np.std(rf_label_pr[:,1]))).T
#prob_matrix_ab_pr=np.vstack(((ab_label_tw[:,1]-np.mean(ab_label_tw[:,1]))/np.std(ab_label_tw[:,1]),(ab_label_pr[:,1]-np.mean(ab_label_pr[:,1]))/np.std(ab_label_pr[:,1]))).T
#prob_matrix_gb_pr=np.vstack(((gb_label_tw[:,1]-np.mean(gb_label_tw[:,1]))/np.std(gb_label_tw[:,1]),(gb_label_pr[:,1]-np.mean(gb_label_pr[:,1]))/np.std(gb_label_pr[:,1]))).T
#prob_matrix_en_pr=np.vstack(((en_label_tw[:,1]-np.mean(en_label_tw[:,1]))/np.std(en_label_tw[:,1]),(en_label_pr[:,1]-np.mean(en_label_pr[:,1]))/np.std(en_label_pr[:,1]))).T
#
##tw vs demos
#prob_matrix_lr_de=np.vstack(((lr_label_tw[:,1]-np.mean(lr_label_tw[:,1]))/np.std(lr_label_tw[:,1]),(lr_label_de[:,1]-np.mean(lr_label_de[:,1]))/np.std(lr_label_de[:,1]))).T
#prob_matrix_rf_de=np.vstack(((rf_label_tw[:,1]-np.mean(rf_label_tw[:,1]))/np.std(rf_label_tw[:,1]),(rf_label_de[:,1]-np.mean(rf_label_de[:,1]))/np.std(rf_label_de[:,1]))).T
#prob_matrix_ab_de=np.vstack(((ab_label_tw[:,1]-np.mean(ab_label_tw[:,1]))/np.std(ab_label_tw[:,1]),(ab_label_de[:,1]-np.mean(ab_label_de[:,1]))/np.std(ab_label_de[:,1]))).T
#prob_matrix_gb_de=np.vstack(((gb_label_tw[:,1]-np.mean(gb_label_tw[:,1]))/np.std(gb_label_tw[:,1]),(gb_label_de[:,1]-np.mean(gb_label_de[:,1]))/np.std(gb_label_de[:,1]))).T
#prob_matrix_en_de=np.vstack(((en_label_tw[:,1]-np.mean(en_label_tw[:,1]))/np.std(en_label_tw[:,1]),(en_label_de[:,1]-np.mean(en_label_de[:,1]))/np.std(en_label_de[:,1]))).T
#
##tw vs drugs
#prob_matrix_lr_dr=np.vstack(((lr_label_tw[:,1]-np.mean(lr_label_tw[:,1]))/np.std(lr_label_tw[:,1]),(lr_label_dr[:,1]-np.mean(lr_label_dr[:,1]))/np.std(lr_label_dr[:,1]))).T
#prob_matrix_rf_dr=np.vstack(((rf_label_tw[:,1]-np.mean(rf_label_tw[:,1]))/np.std(rf_label_tw[:,1]),(rf_label_dr[:,1]-np.mean(rf_label_dr[:,1]))/np.std(rf_label_dr[:,1]))).T
#prob_matrix_ab_dr=np.vstack(((ab_label_tw[:,1]-np.mean(ab_label_tw[:,1]))/np.std(ab_label_tw[:,1]),(ab_label_dr[:,1]-np.mean(ab_label_dr[:,1]))/np.std(ab_label_dr[:,1]))).T
#prob_matrix_gb_dr=np.vstack(((gb_label_tw[:,1]-np.mean(gb_label_tw[:,1]))/np.std(gb_label_tw[:,1]),(gb_label_dr[:,1]-np.mean(gb_label_dr[:,1]))/np.std(gb_label_dr[:,1]))).T
#prob_matrix_en_dr=np.vstack(((en_label_tw[:,1]-np.mean(en_label_tw[:,1]))/np.std(en_label_tw[:,1]),(en_label_dr[:,1]-np.mean(en_label_dr[:,1]))/np.std(en_label_dr[:,1]))).T
#
#print np.mean(prob_matrix_lr_co,axis=0)
#print np.std(prob_matrix_lr_co,axis=0)
#print np.mean(prob_matrix_rf_co,axis=0)
#print np.std(prob_matrix_rf_co,axis=0)
#print np.mean(prob_matrix_ab_co,axis=0)
#print np.std(prob_matrix_ab_co,axis=0)
#print np.mean(prob_matrix_gb_co,axis=0)
#print np.std(prob_matrix_gb_co,axis=0)
#print np.mean(prob_matrix_en_co,axis=0)
#print np.std(prob_matrix_en_co,axis=0)
#print "begin GLM", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#prob_matrix_lr=statsmodels.tools.tools.add_constant(prob_matrix_lr_co, prepend=True,has_constant='add')
#print prob_matrix_lr.shape
#prob_matrix_rf=statsmodels.tools.tools.add_constant(prob_matrix_rf_co, prepend=True,has_constant='add')
##prob_matrix_rf=sm.add_constant(prob_matrix_rf,has_constant='skip')
#print prob_matrix_rf.shape
#prob_matrix_ab=statsmodels.tools.tools.add_constant(prob_matrix_ab_co, prepend=True,has_constant='add')
#print prob_matrix_ab.shape
#prob_matrix_gb=statsmodels.tools.tools.add_constant(prob_matrix_gb_co, prepend=True,has_constant='add')
#print prob_matrix_gb.shape
#prob_matrix_en=statsmodels.tools.tools.add_constant(prob_matrix_en_co, prepend=True,has_constant='add')
#print prob_matrix_en.shape
#bin_model_lr=sm.GLM(test_labels,prob_matrix_lr,family=sm.families.Binomial())
#results_lr=bin_model_lr.fit()
#print("lr",results_lr.summary())
#bin_model_rf=sm.GLM(test_labels,prob_matrix_rf,family=sm.families.Binomial())
#results_rf=bin_model_rf.fit()
#print("rf",results_rf.summary())
#bin_model_ab=sm.GLM(test_labels,prob_matrix_ab,family=sm.families.Binomial())
#results_ab=bin_model_ab.fit()
#print("ab",results_ab.summary())
#bin_model_gb=sm.GLM(test_labels,prob_matrix_gb,family=sm.families.Binomial())
#results_gb=bin_model_gb.fit()
#print("gb",results_gb.summary())
#bin_model_en=sm.GLM(test_labels,prob_matrix_en,family=sm.families.Binomial())
#results_en=bin_model_en.fit()
#print("en",results_en.summary())
#print "done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#
#print np.mean(prob_matrix_lr_pr,axis=0)
#print np.std(prob_matrix_lr_pr,axis=0)
#print np.mean(prob_matrix_rf_pr,axis=0)
#print np.std(prob_matrix_rf_pr,axis=0)
#print np.mean(prob_matrix_ab_pr,axis=0)
#print np.std(prob_matrix_ab_pr,axis=0)
#print np.mean(prob_matrix_gb_pr,axis=0)
#print np.std(prob_matrix_gb_pr,axis=0)
#print np.mean(prob_matrix_en_pr,axis=0)
#print np.std(prob_matrix_en_pr,axis=0)
#print "begin GLM", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#prob_matrix_lr=statsmodels.tools.tools.add_constant(prob_matrix_lr_pr, prepend=True,has_constant='add')
#print prob_matrix_lr.shape
#prob_matrix_rf=statsmodels.tools.tools.add_constant(prob_matrix_rf_pr, prepend=True,has_constant='add')
##prob_matrix_rf=sm.add_constant(prob_matrix_rf,has_constant='skip')
#print prob_matrix_rf.shape
#prob_matrix_ab=statsmodels.tools.tools.add_constant(prob_matrix_ab_pr, prepend=True,has_constant='add')
#print prob_matrix_ab.shape
#prob_matrix_gb=statsmodels.tools.tools.add_constant(prob_matrix_gb_pr, prepend=True,has_constant='add')
#print prob_matrix_gb.shape
#prob_matrix_en=statsmodels.tools.tools.add_constant(prob_matrix_en_pr, prepend=True,has_constant='add')
#print prob_matrix_en.shape
#bin_model_lr=sm.GLM(test_labels,prob_matrix_lr,family=sm.families.Binomial())
#results_lr=bin_model_lr.fit()
#print("lr",results_lr.summary())
#bin_model_rf=sm.GLM(test_labels,prob_matrix_rf,family=sm.families.Binomial())
#results_rf=bin_model_rf.fit()
#print("rf",results_rf.summary())
#bin_model_ab=sm.GLM(test_labels,prob_matrix_ab,family=sm.families.Binomial())
#results_ab=bin_model_ab.fit()
#print("ab",results_ab.summary())
#bin_model_gb=sm.GLM(test_labels,prob_matrix_gb,family=sm.families.Binomial())
#results_gb=bin_model_gb.fit()
#print("gb",results_gb.summary())
#bin_model_en=sm.GLM(test_labels,prob_matrix_en,family=sm.families.Binomial())
#results_en=bin_model_en.fit()
#print("en",results_en.summary())
#print "done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#
#print np.mean(prob_matrix_lr_de,axis=0)
#print np.std(prob_matrix_lr_de,axis=0)
#print np.mean(prob_matrix_rf_de,axis=0)
#print np.std(prob_matrix_rf_de,axis=0)
#print np.mean(prob_matrix_ab_de,axis=0)
#print np.std(prob_matrix_ab_de,axis=0)
#print np.mean(prob_matrix_gb_de,axis=0)
#print np.std(prob_matrix_gb_de,axis=0)
#print np.mean(prob_matrix_en_de,axis=0)
#print np.std(prob_matrix_en_de,axis=0)
#print "begin GLM", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#prob_matrix_lr=statsmodels.tools.tools.add_constant(prob_matrix_lr_de, prepend=True,has_constant='add')
#print prob_matrix_lr.shape
#prob_matrix_rf=statsmodels.tools.tools.add_constant(prob_matrix_rf_de, prepend=True,has_constant='add')
##prob_matrix_rf=sm.add_constant(prob_matrix_rf,has_constant='skip')
#print prob_matrix_rf.shape
#prob_matrix_ab=statsmodels.tools.tools.add_constant(prob_matrix_ab_de, prepend=True,has_constant='add')
#print prob_matrix_ab.shape
#prob_matrix_gb=statsmodels.tools.tools.add_constant(prob_matrix_gb_de, prepend=True,has_constant='add')
#print prob_matrix_gb.shape
#prob_matrix_en=statsmodels.tools.tools.add_constant(prob_matrix_en_de, prepend=True,has_constant='add')
#print prob_matrix_en.shape
#bin_model_lr=sm.GLM(test_labels,prob_matrix_lr,family=sm.families.Binomial())
#results_lr=bin_model_lr.fit()
#print("lr_de",results_lr.summary())
#bin_model_rf=sm.GLM(test_labels,prob_matrix_rf,family=sm.families.Binomial())
#results_rf=bin_model_rf.fit()
#print("rf_de",results_rf.summary())
#bin_model_ab=sm.GLM(test_labels,prob_matrix_ab,family=sm.families.Binomial())
#results_ab=bin_model_ab.fit()
#print("ab_de",results_ab.summary())
#bin_model_gb=sm.GLM(test_labels,prob_matrix_gb,family=sm.families.Binomial())
#results_gb=bin_model_gb.fit()
#print("gb_de",results_gb.summary())
#bin_model_en=sm.GLM(test_labels,prob_matrix_en,family=sm.families.Binomial())
#results_en=bin_model_en.fit()
#print("en_de",results_en.summary())
#print "done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#
#print np.mean(prob_matrix_lr_dr,axis=0)
#print np.std(prob_matrix_lr_dr,axis=0)
#print np.mean(prob_matrix_rf_dr,axis=0)
#print np.std(prob_matrix_rf_dr,axis=0)
#print np.mean(prob_matrix_ab_dr,axis=0)
#print np.std(prob_matrix_ab_dr,axis=0)
#print np.mean(prob_matrix_gb_dr,axis=0)
#print np.std(prob_matrix_gb_dr,axis=0)
#print np.mean(prob_matrix_en_dr,axis=0)
#print np.std(prob_matrix_en_dr,axis=0)
#print "begin GLM", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#prob_matrix_lr=statsmodels.tools.tools.add_constant(prob_matrix_lr_dr, prepend=True,has_constant='add')
#print prob_matrix_lr.shape
#prob_matrix_rf=statsmodels.tools.tools.add_constant(prob_matrix_rf_dr, prepend=True,has_constant='add')
##prob_matrix_rf=sm.add_constant(prob_matrix_rf,has_constant='skip')
#print prob_matrix_rf.shape
#prob_matrix_ab=statsmodels.tools.tools.add_constant(prob_matrix_ab_dr, prepend=True,has_constant='add')
#print prob_matrix_ab.shape
#prob_matrix_gb=statsmodels.tools.tools.add_constant(prob_matrix_gb_dr, prepend=True,has_constant='add')
#print prob_matrix_gb.shape
#prob_matrix_en=statsmodels.tools.tools.add_constant(prob_matrix_en_dr, prepend=True,has_constant='add')
#print prob_matrix_en.shape
#bin_model_lr=sm.GLM(test_labels,prob_matrix_lr,family=sm.families.Binomial())
#results_lr=bin_model_lr.fit()
#print("lr_dr",results_lr.summary())
#bin_model_rf=sm.GLM(test_labels,prob_matrix_rf,family=sm.families.Binomial())
#results_rf=bin_model_rf.fit()
#print("rf_dr",results_rf.summary())
#bin_model_ab=sm.GLM(test_labels,prob_matrix_ab,family=sm.families.Binomial())
#results_ab=bin_model_ab.fit()
#print("ab_dr",results_ab.summary())
#bin_model_gb=sm.GLM(test_labels,prob_matrix_gb,family=sm.families.Binomial())
#results_gb=bin_model_gb.fit()
#print("gb_dr",results_gb.summary())
#bin_model_en=sm.GLM(test_labels,prob_matrix_en,family=sm.families.Binomial())
#results_en=bin_model_en.fit()
#print("en_dr",results_en.summary())
print "done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

