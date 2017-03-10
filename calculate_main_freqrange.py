# -*- coding: utf-8 -*-
"""

Author: shidephen
"""

from librosa import *
import numpy as np
from matplotlib import pyplot as plt
import os


equalizer_freqs = np.array([31., 63., 125., 250., 500., 1000., 2000., 4000., 8000., 16000.])

A = np.array([1.0, -1.99976, 0.999779,
              1.0, -1.99947, 0.999551,
              1.0, -1.99879, 0.99911,
              1.0, -1.99695, 0.998221,
              1.0, -1.99138, 0.996447,
              1.0, -1.97273, 0.992925,
              1.0, -1.90596, 0.986043,
              1.0, -1.66149, 0.973382,
              1.0, -0.816983, 0.955581,
              1.0, 1.27763, 0.96275])
A = A.reshape((10, 3))

B = np.array([0.000110406, 0, -0.000110406,
              0.000224346, 0, -0.000224346,
              0.000445015, 0, -0.000445015,
              0.000889494, 0, -0.000889494,
              0.00177628, 0, -0.00177628,
              0.0035373, 0, -0.0035373,
              0.00697873, 0, -0.00697873,
              0.0133092, 0, -0.0133092,
              0.0222094, 0, -0.0222094,
              0.0186248, 0, -0.0186248])
B = B.reshape((10, 3))


def calc_freq_feature(X, fs, fft_size):
    freq_bins = np.round(equalizer_freqs * fft_size / fs).astype(np.int)
    E_bin = np.array([])
    for i in freq_bins:
        np.append(E_bin, np.mean(X[i-2:i+3]))
    return E_bin


def calc_main_range(path, threshold):
    root, filename = os.path.split(path)
    fft_size = 8192
    x, fs = load(path, None)
    X = stft(x, fft_size)
    freqs = np.round(fft_frequencies(fs, fft_size)).astype(np.int)

    # plot energy figure
    mean_energy = np.mean(np.abs(X), axis=1) ** 2
    peak_pos = np.argmax(mean_energy)

    x_axis = np.arange(len(freqs)) * fs / 2.0 / len(freqs)
    ample = logamplitude(mean_energy, ref_power=np.max)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, ample)
    plt.scatter([freqs[peak_pos], ], [np.max(ample), ], 50, 'blue')
    plt.annotate('(%d, %f)' % (freqs[peak_pos], mean_energy[peak_pos]),
                 xy=(freqs[peak_pos], np.max(ample)), xycoords='data',
                 xytext=(+10, 0), textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.subplot(2, 1, 2)
    display.specshow(logamplitude(X ** 2, ref_power=np.max), fs, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    pic_path = os.path.join(root, filename + '.jpg')
    plt.savefig(pic_path)

    low_f = np.where(freqs <=4000)[0]
    high_f = np.where((8000<=freqs)&(freqs<=16000))[0]

    lowf_E = mean_energy[low_f]
    highf_E = mean_energy[high_f]

    ll, lr = calc_main_freq(lowf_E, threshold)
    hl, hr = calc_main_freq(highf_E, threshold)

    low_main_range = low_f[range(ll, lr)]
    high_main_range = high_f[range(hl, hr)]

    return freqs[low_main_range], freqs[high_main_range], X


def calc_main_freq(mean_energy, threshold):
    total_energy = np.sum(mean_energy)
    peak_pos = np.argmax(mean_energy)

    max_range = len(mean_energy) - peak_pos
    high_b = 0
    for i in range(max_range):
        l = max(0, peak_pos - i)
        energy = np.sum(mean_energy[l:peak_pos+i])
        if (energy/total_energy) >= threshold:
            high_b = i
            break

    return max(0, peak_pos - high_b), peak_pos + high_b


for root, dirs, filenames in os.walk('output'):
    for f in filenames:
        filename = os.path.join(root, f)
        l, h, _ = calc_main_range(filename, 0.99)
        print(l)
        print(h)
