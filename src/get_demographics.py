import numpy as np
import sys
import scipy as sp
from scipy.sparse import csr_matrix
from collections import defaultdict
#case=sys.argv[1]
#control=sys.argv[2]
#filename_mrns='train_mrns_merge_subwDemog_CID_1to1_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv' + case + control + '.npy'
#mrns=np.load(filename_mrns)
#filename_mrn2label='train_mrn2label_merge_subwDemog_CID_1to1_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv' + case + control + '.npy'
#m2l=np.load(filename_mrn2label)
#m2l=m2l[()]
demo_counts=defaultdict(lambda: defaultdict(list))
demo_freqs=defaultdict(lambda: defaultdict(list))
cases=['G','G','G','G','G','T','T','T','T','T','C','C','C','C','C']
controls=['N','I','C','CI','R','N','I','C','CI','R','N','I','C','CI','R']
for i in range(0,15):
    case=cases[i]
    control=controls[i]
    filename_labels='train_labelswCID_1to1_cond_dera_sou_merge_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy'
    labels=np.load(filename_labels)
    filename_e2i='events2colswCID_1to1_cond_dera_sou_merge_raceeth_mergeset_noctrlicd_new_fixcv'+case+control+'.npy'
    e2c=np.load(filename_e2i)
    e2c=e2c[()]
    filename_all='demo_events_merge_subwDemog_wCID_1to1_cond_dera_sou_raceeth_mergeset_noctrlicd_new_fixcv' + case + control + '.npy'
    demo_events=np.load(filename_all)
    filename_mtrain='matrix_trainwCID_1to1_cond_dera_sou_merge_raceeth_mergeset_noctrlicd_new_fixcv' + case + control + '.npz'
    matrix=sp.sparse.load_npz(filename_mtrain)

    demos_count=defaultdict(lambda: defaultdict(int))
    case_len=np.sum(labels)
    ctrl_len=len(labels)-case_len
    print case,control,case_len,ctrl_len
    for t in range(0,len(labels)):
        label=labels[t]  
        for d in demo_events:
	    col=e2c[d]
 	    if d=='American Indian' or d=='Pacific Islander' or d=='Asian':
                demos_count[label]['other']+=matrix[t,col]
	    elif d=='U':
	  	demos_count[label]['Unknown']+=matrix[t,col]
	    else:
		demos_count[label][d]+=matrix[t,col]
    for d in demos_count[1].keys():
        demo_counts[case+'ca'][d].append(demos_count[1][d])
        demo_counts[control+'co'][d].append(demos_count[0][d])
	demo_freqs[case+'ca'][d].append(demos_count[1][d]/float(case_len))
        demo_freqs[control+'co'][d].append(demos_count[0][d]/float(ctrl_len))

for c in demo_counts.keys():
    for d in demo_counts[c]:
	print c,d,demo_counts[c][d],demo_freqs[c][d]
#    for d in demos_count[label].keys():
	#print label,d,str(demos_count[label][d]),str(float(demos_count[label][d])/len(np.argwhere(labels==label)))
    #print str(len(np.argwhere(labels==label)))
