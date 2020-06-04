#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods
#This script is to determine the percentage of EHR patients with tirsch criteria for acute ischemic stroke
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
model=sys.argv[3]
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
#build tw model
pids=np.load({person_ids_EHR_test_filename}+model+'_thresh'+case+control+'.npy')
SQL = '''SELECT distinct person_id, condition_source_concept_id cid, condition_source_value icd, condition_start_date dx_date
        FROM {condition_occurrence table}
        WHERE condition_source_value in (select icd from {tirsch_criteria_table})
        and person_id in %s''' %str(tuple(pids))
c.execute(SQL)
results = c.fetchall()
pids_wtirsch=set()
for mrn, cid, icd, dx_date in results:
    mrn = int(mrn)
    pids_wtirsch.add(mrn)

print len(pids),len(pids_wtirsch),float(len(pids_wtirsch))/len(pids)
