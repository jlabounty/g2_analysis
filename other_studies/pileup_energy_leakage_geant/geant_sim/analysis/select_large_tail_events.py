import ROOT as r
import os
import sys

path_to_lib = '/simulation/'
# cppyy.add_include_path(os.path.join(path_to_lib, 'shared/include/'))
r.gSystem.Load(os.path.join(path_to_lib,"build/shared/libRootDict.so")) 

indir = sys.argv[1]
ch = r.TChain('sim')
ch.Add(os.path.join(indir, '*root'))

ch.CopyTree('calo.Edep')