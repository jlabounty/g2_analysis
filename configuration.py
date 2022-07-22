import os
from pathlib import Path
import sys
# from dataclasses import dataclass, asdict
import json
from typing import List
import tomlkit 

# @dataclass
class AnalysisConfig:
    
    def __init__(self,di):
        self.blinding_phrase = None
        self.infile = None
        if(type(di) == tomlkit.TOMLDocument ):
            self.d = di 
        elif(type(di) in [str, Path] and os.path.exists(di)):
            self.infile = di
            with open(di, 'r') as f:
                self.d = tomlkit.load(f)
        else:
            raise NotImplementedError

    def __getitem__(self, arg):
        if(arg in ['blinding_string', 'blinding_phrase']):
            return self._get_blinding_string()  
        else:
            return self.d[arg]

    def __setitem__(self,arg,val):
        self.d.__setitem__(arg,val)

    def update(self):
        if(self.infile is None):
            raise FileNotFoundError('Please specify which file to update')
        self.dump(self.infile)

    def _get_blinding_string(self):
        # print("getting blinding")
        if(self.blinding_phrase is None):
            # print("ok.1")
            assert 'blinding_file' in self.d
            blinding_file = self['blinding_file']
            assert os.path.exists(blinding_file)
            with open(blinding_file, 'r') as f:
                self.blinding_phrase = f.readlines()[0].strip()
            # print(f'{self.blinding_phrase=}')
        return self.blinding_phrase
        
    def dumps(self, **kwargs):
        return tomlkit.dumps(self.d, **kwargs)

    def dump(self, outfile):
        with open(outfile, 'w') as f:
            tomlkit.dump(self.d, f)

    def get_directory(self):
        return self['directory']

    def get_raw_file(self):
        return os.path.join(self['directory'], self['raw']['file'])

    def get_pileup_file(self):
        if(self['pileup_corr']['complete']):
            return os.path.join(self['directory'], self['pileup_corr']['file'])
        else:
            raise ValueError("Pileup correction not complete!")
