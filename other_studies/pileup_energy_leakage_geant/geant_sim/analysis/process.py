import cppyy
import sys 
import numpy as np 
import hist 
import os
import matplotlib.pyplot as plt
import matplotlib
import hist
import pandas
import time
import pickle
import gzip


# for _ in range(2):
# path_to_lib = '/home/jlab/github/pioneer/straPIONEER/shared'
path_to_lib = '/simulation/'
cppyy.add_include_path(os.path.join(path_to_lib, 'shared/include/'))
cppyy.load_library(os.path.join(path_to_lib,"build/shared/libRootDict.so")) 

def get_edep_calo_tail(event, threshold = 40):
    '''function to get the energy deposited in the gm2 calo xtals and in the tail catcher'''
    deps = {}
    tail_deps = {}
    total = 0
    total_with_threshold = 0
    tail = 0
    init = event.fInit[0].GetMomentum()[2]
    xpos = event.fInit[0].GetPosition()[0]
    ypos = event.fInit[0].GetPosition()[1]
    for x in event.fCalo:
        name = x.GetCaloID()
        # print(name, x.GetTotalEnergyDeposit())
        ei = x.GetTotalEnergyDeposit()
        if(name < 5000):
            total += ei
            if(ei > threshold):
                total_with_threshold += ei
            deps[name] = ei
        else:
            if(ei > 500):
                print(name, init, ei)
            tail += ei
            tail_deps[name] = ei
    di = {
        'edep':total, 
        'tail':tail, 
        'init':init, 
        'x':xpos, 
        'y':ypos, 
        'edep_thresh':total_with_threshold, 
        'threshold':threshold 
    }
    # di.update(deps)
    # di.update(tail_deps)
    return di

def main():
    indir = sys.argv[1]
    files = [os.path.join(indir,x) for x in os.listdir(indir) if '.root' in x]
    all_energies = []
    for j,file in enumerate(files):
        print("Processing file:", file, f'{j+1}/{len(files)}')
        loader = cppyy.gbl.PIMCFileLoader(file)
        for i in range(loader.GetEntries()):
            event = loader.GetFromFile(i)
            all_energies.append(get_edep_calo_tail(event))
            # if(i > 100):
            #     break
        loader.CloseFile()

    # df = pandas.DataFrame(np.array(all_energies), columns=['edep', 'tail', 'init', 'x', 'y', 'edep_thresh', 'threshold'])
    df = pandas.DataFrame(all_energies)
    df.head()

    outfile = f'./dataframe_{time.time()}_{df.shape[0]}.csv'
    df.to_csv(outfile)
    # df.to_pickle(outfile)
    # with gzip.open(outfile, 'wb') as f:
    #     pickle.dump(df,f)

if __name__ == "__main__":
    main()