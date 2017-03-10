
import numpy as np
import os
from autofader import measure_loudness
import matplotlib.pyplot as plt
from librosa import load


for root, dirs, files in os.walk('Freemusic'):
    dr = []
    for filename in files:
        if not filename.endswith('.wav'):
            continue

        filepath = os.path.join(root, filename)
        file_name, ext = filename.split('.')
        Vdb_file = os.path.join(root, file_name+'_Vdb.txt')
        if not os.path.exists(Vdb_file):
            x, fs = load(filepath, None)
            L, peak, Vdb = measure_loudness(x, fs)
            np.savetxt(Vdb_file, Vdb)
        else:
            Vdb = np.loadtxt(Vdb_file)

        d = np.std(Vdb)
        dr.append(d)
        print('%s dynamic range in VdB: %.2f' %(filename, d))

    dr = np.array(dr)
    dr_extreme = int(np.max(dr) - np.min(dr))
    dr_mean = np.mean(dr)
    dr_median = np.median(dr)
    print 'extreme diff: ', dr_extreme
    print 'mean: ', dr_mean
    print 'median: ', dr_median
    plt.figure()
    plt.hist(dr)
    plt.show()
