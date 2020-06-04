#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods
#This script is to train models for each fold and proportion of case controls for the robustness analyses of Supplementary figures 1 and 2
import MySQLdb
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import sys
import csv
import numpy as np
import random
import scipy as sp
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
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
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
print "start", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
def main(train_mat,test_mat,train_labels,test_labels,sample_inds,rep,sz,case,control):
    matrix=sp.sparse.load_npz(train_mat)
    matrix_test=sp.sparse.load_npz(test_mat)
    train_labels=np.load(train_labels)
    test_labels=np.load(test_labels)
    sample_inds=np.load(sample_inds)
    cv_te_metrics=np.zeros(shape=(11,1))
    X_new=matrix.tocsr()[sample_inds,:]
    z=np.argwhere(np.sum(X_new,axis=0) >0)[:,1]
    X_new=X_new[:,z]
    y=train_labels[sample_inds]
    num_feat=len(z)
    cv_te_metrics[10]=num_feat
    matrix_test_samp=matrix_test.tocsr()[:,z]
    #LR model
    C_val=.1
    lr_clf_s=linear_model.LogisticRegression(penalty = 'l1', C = C_val)
    #EN model
    en_clf_s=linear_model.SGDClassifier(penalty='elasticnet',l1_ratio=.01,alpha=.01,loss='log')
    #RF model
    rf_clf_s= ensemble.RandomForestClassifier(n_estimators=1000, max_features="sqrt", max_depth=100, oob_score=True)
    #AB model
    ab_clf_s= ensemble.AdaBoostClassifier(learning_rate=.1,n_estimators=1000)
    #GB model 
    gb_clf_s=ensemble.GradientBoostingClassifier(learning_rate=.1, subsample=.5,n_estimators=1000, max_features="sqrt", max_depth=10)
    lr_test_label=lr_clf_s.predict_proba(matrix_test_samp)
    cv_te_metrics[0]=metrics.roc_auc_score(test_labels, lr_test_label[:,1] )
    cv_te_metrics[1]=metrics.average_precision_score(test_labels,lr_test_label[:,1])
    en_clf_s.fit(X_new,y)
    en_test_label=en_clf_s.predict_proba(matrix_test_samp)
    cv_te_metrics[2]=metrics.roc_auc_score(test_labels, en_test_label[:,1])
    cv_te_metrics[3]=metrics.average_precision_score(test_labels,en_test_label[:,1])
    rf_clf_s.fit(X_new,y)
    rf_test_label=rf_clf_s.predict_proba(matrix_test_samp)
    cv_te_metrics[4]=metrics.roc_auc_score(test_labels, rf_test_label[:,1] )
    cv_te_metrics[5]=metrics.average_precision_score(test_labels,rf_test_label[:,1])
    ab_clf_s.fit(X_new,y)
    ab_test_label=ab_clf_s.predict_proba(matrix_test_samp)
    cv_te_metrics[6]=metrics.roc_auc_score(test_labels, ab_test_label[:,1] )
    cv_te_metrics[7]=metrics.average_precision_score(test_labels,ab_test_label[:,1])
    gb_clf_s.fit(X_new,y)
    gb_test_label=gb_clf_s.predict_proba(matrix_test_samp)
    cv_te_metrics[8]=metrics.roc_auc_score(test_labels, gb_test_label[:,1] )
    cv_te_metrics[9]=metrics.average_precision_score(test_labels,gb_test_label[:,1])
    np.save(cross_val_test_metrics_filename+str(rep)+str(sz)+case+control+".npy",cv_te_metrics)
print "end", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
if __name__=="__main__":
    args = dict([x.split('=') for x in sys.argv[1:]])
    print >> sys.stderr, ""
    print >> sys.stderr, "Running robustness analysis "
    print >> sys.stderr, "-----------------------------------------------------------------------------"
#    print >> sys.stderr, "Summary results will be saved in %(sd)s/%(name)s_solar_strap_results.csv" % args
#    print >> sys.stderr, "Results from each bootstrap will be saved at %(sd)s/%(name)s_solar_strap_allruns.csv" % args
#    print >> sys.stderr, ""

    valid_args = ('train_matrix', 'test_matrix','train_labels','test_labels',
        'sample_inds','rep','sz','case','control')
    for argname in args.keys():
        if not argname in valid_args:
            raise Exception("Provided argument: `%s` is not a valid argument name." % argname)
    main(train_mat = args['train_matrix'],
        test_mat = args['test_matrix'],
        train_labels = args['train_labels'],
        test_labels = args['test_labels'],
        sample_inds = args['sample_inds'],
	rep=args['rep'],
	sz=args['sz'],
 	case=args['case'],
	control=args['control'],
    print "end", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
