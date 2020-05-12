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
database='clinical_gm'
def credential():
        '''import login and passwrd from credential text file'''
        reader=csv.reader(open('mycnf.csv'),delimiter = ",")
        for login, password in reader:
                        login=login
                        passwd=password

        return login, passwd

login,passwd=credential()
train_mrns=np.load('train_mrns_merge_subwDemog_CID_1to1_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy')
print "first entrance to mysql", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
db = MySQLdb.connect(host='127.0.0.1', user ='%s' % (login), passwd='%s' % (passwd), db=database, port=3307)
c = db.cursor()

#SQL ='''select mrn, 0 as label from user_pt2281.rand_mrns_w_cond_NO union select mrn, 1 as label from user_pt2281.STROKE_GOLD_STANDARD_W_DATES_subset_test'''
SQL ='''select distinct person_id,1 from clinical_merge_v5_260318.condition_occurrence where person_id not in %s''' %(str(tuple(train_mrns)))
c.execute(SQL)
results = c.fetchall()

mrns = []
for pid,cid in tqdm(results):
    pid = int(pid)
    mrns.append(pid)
#filename_mrns='train_mrns_v5_subwDemog_wCID' + case + control + '.npy'
#filename_mrns2='mrns_EHR_all'+case+control+'.npy'
#mrns=np.load(filename_mrns)
#mrns=np.concatenate((mrns,np.load(filename_mrns2)),axis=0)
#mrns.tolist()
print len(mrns), datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

#filename_lr='lr_reg_model_pd_v5_subwDemog_wCID_1to1_dera' + case + control + '.sav'
#lr_clf=joblib.load(open(filename_lr,'rb'))
#filename_rf='rf5_model_pd_v5_subwDemog_wCID_1to1_dera' + case + control + '.sav'
#rf_clf=joblib.load(open(filename_rf,'rb'))
#filename_rf10='rf10_model_pd_v5_subwDemog_wCID_1to1_dera' + case + control + '.sav'
#rf_clf10=joblib.load(open(filename_rf10,'rb'))
#filename_all='allPD_events_v5_subwDemog_wCID_1to1_dera' + case + control + '.npy'
#rf_events=np.load(filename_all)
#rf_events.tolist()
cond_name="cond_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv"+case+control+".npy"
cond_events=np.load(cond_name).tolist()
proc_name="proc_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv"+case+control+".npy"
proc_events=np.load(proc_name).tolist()
drug_name="drug_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv"+case+control+".npy"
drug_events=np.load(drug_name).tolist()
demo_name="demo_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv"+case+control+".npy"
demo_events=np.load(demo_name).tolist()
rf_events=cond_events+proc_events+drug_events+demo_events
print len(rf_events)
#filename_ab='ab_model_pd_v5_subwDemog_wCID_1to1_dera' + case + control + '.sav'
#ab_clf=joblib.load(filename_ab)
#filename_en='en_model_pd_v5_subwDemog' + case + control + '.sav'
#en_clf=pickle.load(open(filename_en,'rb'))
print "getting icd9s", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
it=0
lastind=len(mrns)-1
mrn2dx = defaultdict(lambda : defaultdict(int))
mrn_lastdate=defaultdict(lambda:date(1900,1,1))
for mcount in tqdm(range(0,len(mrns))):
    if mcount>0 and mcount % 100000 == 0 or mcount==lastind:
	print mcount, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
	SQL = '''SELECT person_id mrn, condition_concept_id cid, condition_source_value icd, condition_start_date dx_date
        FROM clinical_merge_v5_260318.condition_occurrence
	WHERE person_id in %s
        order by mrn, dx_date''' %str(tuple(mrns[it*100000:mcount]))
#        WHERE condition_source_value not like 'I9:346.3%%'
#        and condition_source_value not like 'I9:431%%'
#        and condition_source_value not like 'I9:340%%'
#        and condition_source_value != 'I9:251.0'
#        and condition_source_value not like 'I9:191%%'
#        and condition_source_value not like 'I9:225%%'
#        and condition_source_value not like 'I9:344%%'
#        and condition_source_value not like 'I9:434.%%'
#        and condition_source_value not like 'I10:I6%.%%%'
#        and condition_source_value not like 'I9:346.6%%'
#        and condition_source_value not like 'I9:430'
#        and condition_source_value not like 'I9:432.%%'
#        and condition_source_value not like 'I9:433.%%'
#        and condition_source_value not like 'I9:436'
	c.execute(SQL)
	results = c.fetchall()
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

#events = sorted(events)
	print len(rf_events), len(mrn2dx), len(mrn_lastdate)
	SQL = '''SELECT person_id mrn, procedure_concept_id pid, procedure_source_value lab, procedure_date dx_date
	        FROM clinical_merge_v5_260318.procedure_occurrence
	        where person_id in %s
	        order by mrn, dx_date''' %str(tuple(mrns[it*100000:mcount]))
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
	        FROM clinical_merge_v5_260318.drug_era
	        where person_id in %s
	        order by mrn, dx_date''' %str(tuple(mrns[it*100000:mcount]))
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
#    mrn2date_dx[mrn][dx_date][lab] = 1.0
#SQL = '''SELECT p.PERSON_SOURCE_VALUE mrn, OBSERVATION_SOURCE_VALUE lab, OBSERVATION_DATE dx_date
#        FROM clinical_cumc.OBSERVATION o
#        JOIN clinical_cumc.PERSON p using (person_id)
#                
#        where person_source_value in %s
#        order by mrn, dx_date''' %str(tuple(mrns))
## print SQL
#c.execute(SQL)
#results = c.fetchall()
#
#for mrn, lab, dx_date in results:
#    mrn = int(mrn)
#    if lab not in events:
#        events.append(lab)
#    mrn2dx[mrn][lab] = 1
#    mrn2date_dx[mrn][dx_date][lab] = 1.0
#mrns=mrn2dx.keys()

        SQL = '''SELECT person_id mrn, gender_source_value gender, year_of_birth year,month_of_birth month, day_of_birth day, race_source_value race, ethnicity_source_value eth,race_concept_id race_concept_id
        FROM clinical_merge_v5_260318.person
        where person_id in %s
        order by mrn''' %str(tuple(mrns[it*100000:mcount]))
        c.execute(SQL)
        results = c.fetchall()
        for mrn, gender,year,month,day,race,eth,race_concept_id in results:
            mrn = int(mrn)
	    if gender in demo_events:
                mrn2dx[mrn][gender] = 1
	    if race in demo_events:
                mrn2dx[mrn][race]=1
	    if eth in demo_events:
                mrn2dx[mrn][eth]=1
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
            birth_year=int(year)
            age=(mrn_lastdate[mrn].year-birth_year)
            if age > 50:
                mrn2dx[mrn]['age']=1
            else:
                mrn2dx[mrn]['age']=0
	it+=1
event2ind=dict([(e, i) for i,e in enumerate(rf_events)])
#labels=[]
print len(mrn2dx.keys()),'len xt_empi', datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

#rows=[]
#it=0
#lastind=len(mrn2dx)-1
matrix_test=lil_matrix((len(mrns), len(rf_events)), dtype=np.dtype(int))
for mcount, mrn in enumerate(mrns):
#    if mcount>1000:
#        break
#    if mcount % 100 == 0:
#	print mcount,
    #row = []
#	labels.append(mrn2label[mrn])
    for icd in rf_events:
        if icd in mrn2dx[mrn]:
	    matrix_test[mcount,event2ind[icd]]=mrn2dx[mrn][icd]	
#	matrix_test[mcount,event2ind[icd]]=mrn2dx[mrn][icd]
            #row.append(mrn2dx[mrn][icd])
    #    else:
            #row.append(0)
#	    matrix_test[mcount,event2ind[icd]]=0
#    row.append(mrn2dx[mrn]['age'])
    #rows.append(row)
#    if mcount>0 and mcount % 100000 == 0 or mcount==lastind:
#	it+=1
#	#matrix_test = csr_matrix(rows)
#	filename_mrns='mrns_EHR_' + str(it) + '_'+case+control+'.npy'
#	np.save(filename_mrns,mrns_order)
#	mrns_order=[]
#	rows=[]
#	lr_test_label=lr_clf.predict_proba(matrix_test)
#	rf_test_label=rf_clf.predict_proba(matrix_test)
#	rf_10_test_label=rf_clf10.predict_proba(matrix_test)
#	ab_test_label=ab_clf.predict_proba(matrix_test)
#	del(matrix_test)
#	filename_lrtest_label= 'lr_test_labels_pd_v5_EHR_' + str(it)+ '_'+ case+control+'.npy'
#	filename_rftest_label='rf_test_labels_pd_v5_EHR_' + str(it)+ '_'+ case+control+'.npy'
#	filename_rf10test_label='rf_10_test_labels_pd_v5_EHR_' + str(it)+ '_'+ case+control+'.npy'	
#	np.save(filename_lrtest_label,lr_test_label)
#	np.save(filename_rftest_label,rf_test_label)
#	np.save(filename_rf10test_label,rf_10_test_label)
#	filename_abtest_label='ab_test_labels_pd_v5_EHR_' + str(it)+ '_'+ case+control+'.npy'
#	np.save(filename_abtest_label,ab_test_label)
#############EHR
filename_mrns='pids_condEHR_wCID_1to1_cond_dera_raceeth_binage_eventlater_mergeset_noctrlicd_new_fixcv' + case+control+'.npy'
np.save(filename_mrns,mrns)
#filename_labels='labels_testwCID_1to1_cond_dera' + case + control +'.npy'
#np.save(filename_labels,labels)
matrix_test=matrix_test.tocsr()
filename_mtest='matrix_condEHR_testwCID_1to1_cond_dera_raceeth_binage_eventlater_mergeset_noctrlicd_new_fixcv' + case + control + '.npz'
sp.sparse.save_npz(filename_mtest,matrix_test)
#lr_test_label=lr_clf.predict_proba(matrix_test)
#rf_test_label=rf_clf.predict_proba(matrix_test)
#rf_10_test_label=rf_clf10.predict_proba(matrix_test)
#ab_test_label=ab_clf.predict_proba(matrix_test)
##filename_lrtest_label= 'lr_test_labels_pd_v5_EHR_' +  case+control+'.npy'
##filename_rftest_label='rf5_test_labels_pd_v5_EHR_' +  case+control+'.npy'
##filename_rf10test_label='rf10_test_labels_pd_v5_EHR_' + case+control+'.npy'
#filename_lrtest_label= 'lr_test_labels_pd_v5_wCID_1to1_dera' +  case+control+'.npy'
#filename_rf5test_label='rf5_test_labels_pd_v5_wCID_1to1_dera' +  case+control+'.npy'
#filename_rf10test_label='rf10_test_labels_pd_v5_wCID_1to1_dera' + case+control+'.npy'
#np.save(filename_lrtest_label,lr_test_label)
#np.save(filename_rf5test_label,rf_test_label)
#np.save(filename_rf10test_label,rf_10_test_label)
##filename_abtest_label='ab_test_labels_pd_v5_EHR_'+ case+control+'.npy'
#filename_abtest_label='ab_test_labels_pd_v5_wCID_1to1_dera'+ case+control+'.npy'
#np.save(filename_abtest_label,ab_test_label)
print '\n', len(mrn2dx)


#matrix_arr = csr_matrix(rows)
#    notnull_lr=1
#    notnull_rf=1
#    numevents=len(mrn2dx[mrn])
#    for event in mrn2dx[mrn]:
#	if event in rf_events:
#	    notnull_rf=0
#	    if event in lr_events:
#	        notnull_lr=0
#	    rf_rows.append(rfcount)
#	    rf_cols.append(event2ind[event])
#	    rf_values.append(1)
#	    lr_rows.append(lrcount)
#            lr_cols.append(event2ind[event])
#            lr_values.append(1)
#    if notnull_rf==0 and notnull_lr==0:
#	rf_test_mrns.append(mrn)
#        lr_test_mrns.append(mrn)
#	rfcount+=1
#	lrcount+=1
#    elif notnull_lr==1:
#	if notnull_rf==0:
#	    rf_test_mrns.append(mrn)
#	    rfcount+=1
#	for i in range(0,numevents):
#		lr_rows.pop()
#		lr_cols.pop()
#		lr_values.pop()
#print "making csr_matrix(lr_matrix)", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#lr_matrix_test = csr_matrix((lr_values,(lr_rows,lr_cols)),shape=(len(lr_test_mrns),len(rf_events)))
print "making csr_matrix", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#matrix_test = csr_matrix(rows)
print "done with csr", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#print matrix_test.shape
#print lr_matrix_test.shape
#sp.sparse.save_npz("matrix_test_stroke_gm.npz",matrix_test)
##matrix_test=sp.sparse.load_npz("matrix_test.npz")
#filename_mrns= 'mrns_stroke_gm' + case + control + '.npy'
#np.save(filename_mrns,mrns_order)
#lr_test_label=lr_clf.predict_proba(matrix_test)
#filename_lrtest_label= 'lr_test_labels_pd_v5_EHR' + case+control+'.npy'
#filename_rftest_label='rf_test_labels_pd_v5_EHR'+ case+control+'.npy'
#filename_rf10test_label='rf_10_test_labels_pd_v5_EHR'+ case+control+'.npy'
#rf_10_test_label=rf_clf10.predict_proba(matrix_test)
##filename_m2l='mrn2label_EHR.npy'
##np.save(filename_m2l,mrn2label)
#
#np.save(filename_lrtest_label,lr_test_label)
#rf_test_label=rf_clf.predict_proba(matrix_test)
#np.save(filename_rftest_label,rf_test_label)
#np.save(filename_rf10test_label,rf_10_test_label)
#ab_test_label=ab_clf.predict_proba(matrix_test)
#filename_abtest_label='ab_test_labels_pd_v5_EHR'+ case+control+'.npy'
#np.save(filename_abtest_label,ab_test_label)
print "done", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
#en_test_label=ab_clf.predict_proba(matrix_test)
#filename_entest_label='en_test_labels_pd_v5_subwDemog_newmod'+ case+control+'.npy'
#np.save(filename_entest_label,en_test_label)

#SQL='''select t.individual_id from clinical_relationships.columbia_family_id t where t.family_id in (select family_id, count(distinct individual_id) as fam_ascert from clinical_relationships.columbia_family_id where individual_id in %s group by family_id order by fam_ascert DESC limit 10000) c''' %str(tuple(lr_test_empis))
#c.execute(SQL)
#results = c.fetchall()
#lr_test_empis_topfam=[]
#for individual_id in results:
#	lr_test_empis_topfam.append(individual_id)
#
#SQL='''select t.individual_id from clinical_relationships.columbia_family_id t where t.family_id in (select family_id, count(distinct individual_id) as fam_ascert from clinical_relationships.columbia_family_id where individual_id in %s group by family_id order by fam_ascert DESC limit 10000) c''' %str(tuple(rf_test_empis))
#c.execute(SQL)
#results = c.fetchall()
#
#rf_test_empis_topfam=[]
#for individual_id in results:
#        rf_test_empis_topfam.append(individual_id)

#print "writing results", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
##filename_lrresult='lr_reg_pd_v5_trait_EHRsubF_' +case+control+'.txt'
#filename_lrresult='lr_reg_pd_v5_trait_subwDemog' +case+control+'.txt'
#filename_rfresult='rf_pd_v5_trait_subwDemog' +case+control+'.txt'
##filename_rfresult='rf_pd_v5_trait_EHRsubF_' +case+control+'.txt'
#filename_abresult='ab_pd_v5_trait_subwDemog' +case+control+'.txt'
#lr_result=open(filename_lrresult,'w')
#rf_result=open(filename_rfresult,'w')
#ab_result=open(filename_abresult,'w')
#for i in range(0,len(lr_test_empis)):
#	lr_result.write(str(lr_test_empis[i])+'\t434\t'+str(np.log(lr_test_label[i,1]))+'\n')
#	#lr_result.write(str(lr_test_label[i,1])+'\n')
#for i in range(0,len(rf_test_empis)):
#	rf_result.write(str(rf_test_empis[i])+'\t434\t'+str(np.log(rf_test_label[i,1]))+'\n')
#	#rf_result.write(str(rf_test_label[i,1])+'\n')
#	ab_result.write(str(rf_test_empis[i])+'\t434\t'+str(np.log(ab_test_label[i,1]))+'\n')   
#        #rf_result.write(str(rf_test_label[i,1])+'\n')
#
#lr_result.close()
#rf_result.close()
#ab_result.close()
c.close()
db.commit()
db.close()

