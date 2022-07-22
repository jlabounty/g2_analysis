import os 
import sys
from g2_analysis import configuration

dataset = sys.argv[1].strip()
basepath = '/home/jlab/github/g2_analysis/datasets'
subfolders = [
    'submission_scripts',
    'data_raw',
    'data_pileupCorr_noResidGain',
    'data_pileupCorr',
    'plots',
    'scans',
    'fits'
]

thispath = os.path.join(basepath, dataset)
if(os.path.exists(thispath)):
    raise FileExistsError("ERROR: this dataset name already exists")

basefcl = '/home/jlab/github/g2_analysis/scripts/base.fcl'
basesubmit = '/home/jlab/github/g2_analysis/scripts/submit.sh_base'
for x in subfolders:
    pathi = os.path.join(thispath,x)
    os.system(f'mkdir -p {pathi}')
    if('submission_scripts' in x):
        thisfcl = os.path.join(pathi, f"{dataset}.fcl")
        os.system(f'cp {basefcl} {thisfcl}')
        with open(basesubmit, 'r') as fin:
            with open(os.path.join(pathi,'submit.sh'), 'w') as fout:
                for line in fin:
                    fout.write(line.replace('FCL', thisfcl).replace("DATASET", dataset))


# os.system(f'touch {os.path.join(basepath, notes.md)')
with open(os.path.join(thispath, f'notes_{dataset}.md'), 'w') as f:
    f.write(f'# Dataset: {dataset}\n\n---\n')

default_config = '/home/jlab/github/g2_analysis/scripts/default_config.toml'
this_config = configuration.AnalysisConfig(default_config)
this_config['directory'] = thispath 
this_config['dataset'] = dataset

this_config.dump(os.path.join(thispath, f'config_{dataset}.toml'))

print("Dataset is all set up:", thispath)
