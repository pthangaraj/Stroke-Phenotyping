#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods
#This script is to build matrices of performance metrics for the robustness analysis and supplementary figures 1 and 2
import numpy as np
import os.path
cases=['S','C','T']
controls=['C','I','CI','N','R']
pct=['1','0.9','0.7','0.5','0.3','0.1','0.05','0.01','0.005','0.001']
cross_val_testing_set_by samplingpct_filename=
testing_set_metrics_filename=
testing_set_number_folds_filename=

for case in cases:
    for control in controls:
        cv_te_metrics=np.zeros(shape=(10,10*10))
        num_features=np.zeros(shape=(10,10))
        for i in range(0,10):
            for j in range(0,10):
                cv_filename=cross_val_testing_set_by samplingpct_filename+str(j)+str(pct[i])+case+control+".npy"
		if os.path.isfile(cv_filename):
                    arr=np.load(cv_filename)
                    cv_te_metrics[:,i*10+j]=arr[0:10,0]
                    num_features[j,i]=arr[10,0]
		else:
	            print case,control,str(j),str(i)
		    cv_te_metrics[:,i*10+j]=np.zeros((10,))
		    num_features[j,i]=0
        filename_te=testing_set_metrics_filename+case+control+".npy"
        np.save(filename_te,cv_te_metrics)
        filename_fe=testing_set_number_folds_filename+case+control+".npy"
        np.save(filename_fe,num_features)

