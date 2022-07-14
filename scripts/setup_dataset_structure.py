from email.mime import base
import os 
import sys

dataset = sys.argv[1]
basepath = '/home/jlab/github/g2_analysis/datasets'
subfolders = [
    'data_raw',
    'data_pileupCorr_noResidGain',
    'data_pileupCorr',
    'plots/'
]

thispath = os.path.join(basepath, dataset)
if(os.path.exists(thispath)):
    raise FileExistsError("ERROR: this dataset name already exists")

for x in subfolders:
    os.system(f'mkdir -p {os.path.join(basepath,x)}')
