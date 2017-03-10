# -*- coding: utf-8 -*-
"""
自动响度框架
Author: shidephen
"""
import numpy as np
from scipy.io import wavfile
import os
from scipy.signal import lfilter
import sys
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from librosa import *

eps = np.finfo(np.float32).eps
USE_FILTER = False
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

def loudness_filter(x):
    '''
    某响度加权的滤波器
    :param x:
    :return:
    '''
    # Filter 1
    B = [1.176506, -2.353012, 1.176506]
    A = [1, -1.960601, 0.961086]
    x_filtered = lfilter(B, A, x)
    # Filter 2
    B = [0.951539, -1.746297, 0.845694]
    A = [1, -1.746297, 0.797233]
    x_filtered = lfilter(B, A, x_filtered)
    # Filter 3
    B = [1.032534, -1.42493, 0.601922]
    A = [1, -1.42493, 0.634455]
    x_filtered = lfilter(B, A, x_filtered)
    # Filter 4
    B = [0.546949, -0.189981, 0.349394]
    A = [1, -0.189981, -0.103657]
    x_filtered = lfilter(B, A, x_filtered)

    return x_filtered


def calc_mask_by_fft(masker, maskee, fs, coff=0):
    N = 4096
    masker_mX = logamplitude(np.abs(stft(masker, N)) ** 2, ref_power=np.max)
    maskee_mX = logamplitude(np.abs(stft(maskee, N)) ** 2, ref_power=np.max)
    freq_bins = np.round(equalizer_freqs * N / fs).astype(np.int)

    masker_mX = np.mean(masker_mX, axis=1)
    maskee_mX = np.mean(maskee_mX, axis=1)

    masker_rank = np.searchsorted(np.sort(masker_mX), masker_mX)
    maskee_rank = np.searchsorted(np.sort(maskee_mX), maskee_mX)

    mask =  np.where(maskee_rank[freq_bins] > masker_rank[freq_bins],
                    (maskee_mX[freq_bins] - masker_mX[freq_bins]),
                    np.zeros(len(freq_bins)))

    return -np.exp2(-coff) * mask / np.max(mask) * 12.0


def calc_eq_by_mask(tracks, a=0.0):
    N = len(tracks)
    M = len(A)
    rank_threshold = 4.5

    # Calculate frequency magnitude and ranks per track per band
    mags = np.zeros((N, M))
    ranks = np.zeros(mags.shape)
    masks = np.zeros((N, N, M))
    for i in range(N):
        mags[i] = calc_loudness_perbands(tracks[i])
        ranks[i] = np.searchsorted(np.sort(mags[i]), mags[i])

    # Masking detection
    for i in range(N):
        # Masker iteration
        for j in range(N):
            # Maskee iteration
            if j == i:
                continue

            # masks[i, j, :] = mags[i] - mags[j]
            #"""
            masks[i, j, :] = np.where((ranks[i] >rank_threshold) & (rank_threshold > ranks[j]),
                                      mags[i] - mags[j],
                                      np.zeros(M))
            #"""
    np.clip(masks, -12.0, 12.0)
    # np.savetxt('mask.txt', masks.reshape((1,)))

    # Masking selection
    cv = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            # cv[i, j] = np.max(masks[i, :, j])
            cv[i, j] = np.mean(masks[i, :, j])

    return -np.exp2(-a) * cv / np.max(np.abs(cv)) * 12.0, masks


def calc_mask_by_filterbank(masker, maskee, coff=0):
    masker_mX = calc_loudness_perbands(masker)
    maskee_mX = calc_loudness_perbands(maskee)

    masker_rank = np.searchsorted(np.sort(masker_mX), masker_mX)
    maskee_rank = np.searchsorted(np.sort(maskee_mX), maskee_mX)

    mask = np.where(maskee_rank > masker_rank,
                    (maskee_mX - masker_mX),
                    np.zeros(len(maskee_mX)))

    return -np.exp2(-coff) * mask / np.max(mask) * 12.0


def calc_loudness_perbands(x):
    M = len(A)
    loudness = np.zeros(M)
    for i in range(M):
        x_filtered = lfilter(B[i], A[i], x)
        mx = logamplitude(x_filtered**2, ref_power=np.max)
        hists, edges = np.histogram(mx, 100, range=(-75, 0), density=True)
        loudness[i] = np.dot(hists, (edges[1:]+edges[:-1])/2)

    return loudness


def calc_eq_by_avg(tracks, a=0.5):
    N = len(tracks)
    M = len(A)
    fv = np.zeros((N, M))
    for i in range(N):
        x = tracks[i]
        fv[i, :] = calc_loudness_perbands(x)

    l = np.mean(fv)
    cv = 10*np.log10(l / fv)
    coff = np.exp2(a * np.arange(-N, 0))

    for i in range(N):
        cv[i] = cv[i] * coff[i]

    return np.clip(cv, -12.0, 12.0)


def measure_loudness(x, fs, tao=0.035, k=0.85, use_filter=False):
    """
    计算响度特征
    :param x:
    :param fs: 采样率
    :param tao: 时间平滑参数
    :param k: 加权指数
    :return:(响度, 峰值, VdB)
    """

    # N = fs / 4
    N = 4096
    c = np.exp(-1.0 / tao / fs)

    if use_filter:
        x_filtered = loudness_filter(x)  # weighted loudness
    else:
        x_filtered = x  # Flat unit

    Vms = np.zeros(len(x_filtered))
    Vms[0] = (1 - c) * (x_filtered[0] ** 2)

    for i in range(1, len(Vms)):
        Vms[i] = c * Vms[i - 1] + (1 - c) * (x_filtered[i] ** 2)

    Vrms = np.sqrt(Vms[np.arange(N - 1, len(Vms), N)]) + eps
    VdB = 20.0 * np.log10(Vrms)
    ui = np.exp2(-VdB * np.log2(k))
    ui_sum = np.sum(ui)
    wi = ui / ui_sum

    L = np.sum(wi * VdB)
    return L, 20 * np.log10(np.max(x)), VdB



def calc_fader(loudness, peaks, gains):
    '''
    计算推子值
    :param loudness: m个轨道的响度值
    :param peaks: m个轨道的峰值
    :param gains: m个轨道的压缩器补偿
    :return:
    '''
    assert (len(loudness) == len(peaks))
    # max_loudness = np.max(loudness)
    # normalized_loudness = loudness / max_loudness
    gained_loudness = loudness + gains
    mediam_L = np.median(gained_loudness)
    # avg_L = np.mean(loudness)
    Lva = mediam_L - gained_loudness

    for i in range(len(gained_loudness)):
        abs_peak = np.abs(peaks[i])
        Lva[i] = Lva[i] if Lva[i] < abs_peak else abs_peak

    # fv_dB = 20 * np.log10(cv)
    # maximaus_fv = fv_dB - np.max(fv_dB)
    return np.round(Lva, 2)


def calc_compressor(peak, VdB, varian=8):
    '''
    计算压缩器参数
    :param peak: 峰值
    :param VdB: 响度计算中的VdB
    :param varian: 动态方差
    :return: (阈值,压缩比,补偿)
    '''
    profile_threshold = 0.99
    # ratio_threshold = 0.5
    hist, bins = np.histogram(VdB, 100)

    hist = hist / float(sum(hist))
    hist_cum = np.cumsum(hist)
    threshold = np.min(bins[hist_cum >= profile_threshold]) + 3

    # var_origin = np.std(VdB)
    ratio = np.std(VdB) / varian
    if ratio < 1:
        ratio = 1

    gain = (peak - threshold) * (1 - 1.0 / ratio)

    return np.round(threshold, 2), np.round(ratio, 2), np.round(gain, 2)


def automixing(project):
    '''
    自动混音主流程
    :param project:
    :return:
    '''
    filepaths = []
    for root, dirs, files in os.walk(project):
        for file in files:
            if file.endswith('.wav'):
                filepaths.append(os.path.join(root, file))

    m = len(filepaths)
    loudness = np.zeros(m)
    peaks = np.zeros(m)
    filenames = []
    compressor_values = np.zeros((m, 3))

    # find music
    bk_music = None
    for i in range(m):
        if filepaths[i].startswith('music_'):
            bk_music = filepaths[i]
            break

    if bk_music is not None:
        filepaths.remove(bk_music)
        filepaths.insert(0, bk_music)

    # calculate loudness and compressor params
    ref_DR = 8
    for i in range(m):
        filename = os.path.split(filepaths[i])[-1]
        x, fs = load(filepaths[i], None)
        if i == 0:
            loudness[i], peaks[i], VdB = measure_loudness(x, fs, use_filter=False)
            ref_DR = np.std(VdB)
        else:
            loudness[i], peaks[i], VdB = measure_loudness(x, fs, use_filter=USE_FILTER)

        threshold, ratio, gain = calc_compressor(peaks[i], VdB, ref_DR)
        compressor_values[i, 0] = threshold
        compressor_values[i, 1] = ratio
        compressor_values[i, 2] = gain

        print('%s @ Loudness: %f dB; peak: %f' % (filename, loudness[i], peaks[i]))
        print('%s @ Threshold: %.2f dB; Ratio: %.2f; Gain: %.2f' % (filename, threshold, ratio, gain))

        filenames.append(filename)

    faders = calc_fader(loudness, peaks, compressor_values[:, 2])
    max_fades = np.max(faders)

    # make clip
    if max_fades > 0:
        faders = faders - max_fades

    print('Faders: %s' % str(faders))

    return faders, filenames, compressor_values


def write_compressor_chunk(filename, params):
    '''
    保存Waves RCompressor插件配置
    :param filename: 文件名
    :param params: 压缩器参数
    :return:
    '''
    preset = ET.ElementTree(file='RComp_template.xml')  # 参数解释看xml文件
    param_node = preset.find('Preset/PresetData[@Setup=\'SETUP_A\']/Parameters')
    param_value = '160.31 %.1f %.1f * * * * * 10 %.1f 57 10 0 0 149.62 120 0 0 1 0.4 0 1.5 1 * *' \
                  % (params[0], params[2], params[1])

    param_node.text = param_value
    preset.write(filename)


def read_compressor_chunk(filename):
    '''
    从XML中读取RCompressor压缩器参数
    :param filename:
    :return:(Threshold, Ratio, Gain)
    '''
    preset = ET.ElementTree(file=filename)
    param_node = preset.find('Preset/PresetData[@Setup=\'SETUP_A\']/Parameters')
    params = param_node.text.split()

    threshold   = float(params[1])
    gain        = float(params[2])
    ratio       = float(params[9])

    return threshold, ratio, gain


def write_equalizer_chunk(filename, params):
    '''
    保存Waves API-560插件配置
    :param filename: 文件名
    :param params: 10段EQ值，按照升序排列
    :return:
    '''
    analog = False # 是否激活模拟噪声
    preset = ET.ElementTree(file='API560_template.xml')  # 参数解释看xml文件
    param_node = preset.find('Preset/PresetData[@Setup=\'SETUP_A\']/Parameters')
    param_value = '0 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %d * 1 1 * * * * * 0 0 10' \
                  % (params[1], params[3], params[5], params[7], params[9],
                     params[0], params[2], params[4], params[6], params[8],
                     int(analog))

    param_node.text = param_value
    preset.write(filename)


def read_equalizer_chunk(filename):
    '''
    从XML中读取API-560的10段均衡值(升序)
    :param filename:
    :return: np array
    '''
    preset = ET.ElementTree(file=filename)
    param_node = preset.find('Preset/PresetData[@Setup=\'SETUP_A\']/Parameters')
    params = np.array(param_node.text.split())
    gains = params[[6, 1, 7, 2, 8, 3, 9, 4, 10, 5]]

    return gains.astype(np.float)


def save2file(root, filenames, **kwargs):
    m = len(filenames)
    fv = kwargs['fv']
    cv = kwargs['cv']
    target = os.path.join(root, 'fader.txt')
    with open(target, 'w') as f:
        for i in range(m):
            f.writelines(
                '%s @ Fader: %.2f|Threshold: %.1f|Ratio: %.1f|Gain: %.1f\n' \
                % (filenames[i], fv[i], cv[i][0], cv[i][1], cv[i][2]))

            write_compressor_chunk(os.path.join(root, filenames[i].replace('.wav', '_compressor.xps')),
                                   cv[i])
        f.flush()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(-1)

    root = sys.argv[1]
    # root = 'project'
    faders, files, cv = automixing(root)

    save2file(root, files, fv=faders, cv=cv)
