#By Phyllis Thangaraj (pt2281@columbia.edu), Nicholas Tatonetti Lab at Columbia University Irving Medical Center
#Part of manuscript: "Comparative analysis, applications, and interpretation of electronic health record-based stroke phenotyping methods" 
#This script prints out the demographics of all of the models' training sets
import numpy as np
import sys
import scipy as sp
from scipy.sparse import csr_matrix
from collections import defaultdict
demo_counts=defaultdict(lambda: defaultdict(list))
demo_freqs=defaultdict(lambda: defaultdict(list))
cases=['G','G','G','G','G','T','T','T','T','T','C','C','C','C','C']
controls=['N','I','C','CI','R','N','I','C','CI','R','N','I','C','CI','R']
for i in range(0,15):
    case=cases[i]
    control=controls[i]
    filename_labels={training_set labels filename}+case+control+'.npy'
    labels=np.load(filename_labels)
    filename_e2i={events2cols filename}+case+control+'.npy'
    e2c=np.load(filename_e2i)
    e2c=e2c[()]
    filename_all={demographics events filenames} + case + control + '.npy'
    demo_events=np.load(filename_all)
    filename_mtrain={training_set sparse matrix filename} + case + control + '.npz'
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
