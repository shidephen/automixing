# -*- coding: utf-8 -*-
"""

Author: shidephen
"""

from autofader import calc_eq_by_mask, write_equalizer_chunk
from librosa import load
import os, sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(-1)

    tracks = []
    filenames = []
    for root, dirs, files in os.walk(sys.argv[1]):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                x, fs = load(filename, None)
                filenames.append(filename)
                tracks.append(x)

    m = len(tracks)
    cv, mt = calc_eq_by_mask(tracks)
    print('Equalizer atten:')
    print(cv)

    for i in range(m):
        params = cv[i]
        conf = filenames[i].replace('.wav', '_eq.xps')
        write_equalizer_chunk(conf, params)
