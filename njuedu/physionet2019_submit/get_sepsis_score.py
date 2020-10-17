# -*- coding: UTF-8 -*-
# !/usr/bin/python

import numpy as np
import joblib
import os, sys

x_mean = np.array([
    83.8996, 97.0520, 36.8055, 126.2240, 86.2907,
    66.2070, 18.7280, 33.7373, -3.1923, 22.5352,
    0.4597, 7.3889, 39.5049, 96.8883, 103.4265,
    22.4952, 87.5214, 7.7210, 106.1982, 1.5961,
    0.6943, 131.5327, 2.0262, 2.0509, 3.5130,
    4.0541, 1.3423, 5.2734, 32.1134, 10.5383,
    38.9974, 10.5585, 286.5404, 198.6777,
    60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])

x_std = np.array([
    17.6494, 3.0163, 0.6895, 24.2988, 16.6459,
    14.0771, 4.7035, 11.0158, 3.7845, 3.1567,
    6.2684, 0.0710, 9.1087, 3.3971, 430.3638,
    19.0690, 81.7152, 2.3992, 4.9761, 2.0648,
    1.9926, 45.4816, 1.6008, 0.3793, 1.3092,
    0.5844, 2.5511, 20.4142, 6.4362, 2.2302,
    29.8928, 7.0606, 137.3886, 96.8997,
    16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

mFMax = []
fSum_pre = []
fmax = []
fmin = []
grad_temp = []
hess_temp = []
All_grad1 = []
All_grad12 = []
All_grad24 = []


def get_sepsis_score(feature, model):
    feature = genFeature(feature)
    feature[:, 0:34] = imputer_missing_mean_numpy(feature[:, 0:34])
    feature[:, 34:] = imputer_missing_median_numpy(feature[:, 34:])
    # generate predictions
    label = model.predict(feature)
    prob = model.predict_proba(feature)
    # print(label, prob)
    # pb = 0.1
    # # print(prob)
    # if (prob[0][1] > pb):
    #     label = 1
    # else:
    #     label = 0
    return prob[0][1], label


def load_sepsis_model():
    clf = joblib.load('EasyEnsembleLightGBM.pkl')
    return clf


def imputer_missing_mean(testFtr):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imr = joblib.load(os.path.join(BASE_DIR, 'imputer_mean.pkl'), 'wb')
    testFtr = imr.transform(testFtr)
    return testFtr


def imputer_missing_median(testFtr):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imr = joblib.load(os.path.join(BASE_DIR, 'imputer_median.pkl'), 'wb')
    return testFtr


def imputer_missing_mean_numpy(testFtr):
    imr = np.load('imputer_mean_numpy.npy')
    h, w = testFtr.shape
    for i in range(h):
        for j in range(w):
            if np.isnan(testFtr[i, j]):
                testFtr[i, j] = imr[j]
    return testFtr


def imputer_missing_median_numpy(testFtr):
    imr = np.load('imputer_median_numpy.npy')
    h, w = testFtr.shape
    for i in range(h):
        for j in range(w):
            if np.isnan(testFtr[i, j]):
                testFtr[i, j] = imr[j]
    return testFtr


# 输入所有数据特征
def genFeature(data):
    global grad_temp, hess_temp, All_grad1, All_grad12, All_grad24
    exlen = 16
    # feature = data[:, :-1]
    feature = data
    h, w = feature.shape

    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = searchNearValue(i, feature[:, j], 3, True)

    for j in range(w):
        for i in range(h):
            if np.isnan(feature[i, j]):
                feature[i, j] = x_mean[j]

    norm = data_norm(feature)
    res = residual_value(feature)
    grad1 = Grad1(res[:, :exlen])

    grad12 = Grad12(res[:, :exlen])

    grad24 = Grad24(res[:, :exlen])

    grad = np.hstack((grad1, grad12, grad24))

    if h == 1:
        All_grad1 = []
        All_grad12 = []
        All_grad24 = []
        grad_temp = []
        grad_temp.append(grad)
    else:
        grad_temp.append(grad)

    # print(grad)
    temp = np.array(grad_temp)
    h, w = temp.shape
    for j in range(w):
        for i in range(h):
            if np.isnan(temp[i, j]):
                temp[i, j] = searchNearValue(i, temp[:, j], 3, True)

    grad = temp[-1, :]
    grad = np.reshape(grad, (1, w))
    # print("after", grad)
    All_grad1.append(grad[0, :exlen])
    All_grad12.append(grad[0, exlen:2 * exlen])
    All_grad24.append(grad[0, 2 * exlen:3 * exlen])
    hess1 = Grad1(np.array(All_grad1))
    hess12 = Grad12(np.array(All_grad12))
    hess24 = Grad24(np.array(All_grad24))
    hess = np.hstack((hess1, hess12, hess24))
    if h == 1:
        hess_temp = []
        hess_temp.append(hess)
    else:
        hess_temp.append(hess)

    # print(grad)
    temp = np.array(hess_temp)
    h, w = temp.shape
    for j in range(w):
        for i in range(h):
            if np.isnan(temp[i, j]):
                temp[i, j] = searchNearValue(i, temp[:, j], 3, True)

    hess = temp[-1, :]
    hess = np.reshape(hess, (1, w))
    # print("after", grad)

    mutation = mFactor(res[:, :exlen])
    mutationMax = mFactorMax(res[:, :exlen])

    fSum = f_sum(res[:, :exlen])
    fSum8 = f_sum8h(res[:, :exlen])
    fMax = f_max(res[:, :exlen])
    fMin = f_min(res[:, :exlen])
    fMean = f_mean(res[:, :exlen])
    fMedian = f_median(res[:, :exlen])

    fCov = cov_filter(feature[:, :exlen])
    kernel = np.array([[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [4, 8, 12, 8, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    fCov2 = filter2d(feature[:, :exlen], kernel)

    # print(feature[h - 1:h, :].shape)
    # print(grad.shape)
    # print(hess.shape)
    # print(fMedian.shape)
    # print(fCov.shape)
    # print(fCov2.shape)
    # print(res[h - 1:h, :].shape)
    # print(norm.shape)
    # print(fMean.shape)
    f = np.hstack((feature[h - 1:h, :], norm, res[h - 1:h, :], grad, hess, mutation, mutationMax, fSum, fSum8,
                   fMax, fMin, fMean, fMedian, fCov, fCov2))
    # f = np.hstack((feature[h - 1:h, :], grad, mutation, mutationMax, fSum, fSum8,
    #                fMax, fMin, fMean))
    return f


def searchNearValue(index, list, range, isTrain=False):
    indexL = index
    indexH = index
    while indexL >= max(index - range, 0) and indexH < min(index + range, len(list)):
        if np.isnan(list[indexL]) == False:
            return list[indexL]
        if isTrain:
            if np.isnan(list[indexH]) == False:
                return list[indexH]
            else:
                indexH = indexH + 1
        indexL = indexL - 1
    return list[index]


def data_norm(data):
    norm = (data[-1] - x_mean) / x_std
    return np.reshape(norm, (1, len(norm)))


def Grad1(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h > 1:
        grad = data[-1, :] - data[-2, :]
    return grad


def Grad12(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h >= 13:
        grad = data[-1, :] - data[-13, :]
    elif h >= 7:
        grad = data[-1, :] - data[0, :]

    return grad


def Grad24(data):
    h, w = data.shape
    grad = np.zeros(w)
    grad[:] = np.nan
    if h >= 25:
        grad = data[-1, :] - data[-25, :]
    elif h >= 16:
        grad = data[-1, :] - data[0, :]
    return grad


def mFCac(data):
    h, w = data.shape
    m_t = np.nanmean(data, axis=0)
    s_t = np.nanstd(data, axis=0)
    for i in range(w):
        if np.isnan(m_t[i]):
            # m_t[i] = x_mean[i]
            m_t[i] = 0.001
        if np.isnan(s_t[i]):
            s_t[i] = x_std[i]
        if m_t[i] < 0.001 and m_t[i] >= 0:
            m_t[i] = 0.001
        elif m_t[i] > -0.001 and m_t[i] < 0:
            m_t[i] = -0.001
    return np.divide(s_t, m_t)


def mFactor(data):
    h, w = data.shape
    mF = np.zeros((1, w))
    mF[0, :] = np.nan
    if h > 1:
        mF[0, :] = mFCac(data)
    return mF


def mFactorMax(data):
    global mFMax
    h, w = data.shape
    mF = np.zeros((1, w))
    mF[:] = np.nan
    if h == 1:
        mFMax = []
    if h > 11:
        mF = mFCac(data[h - 12:h, :])
        if len(mFMax) >= 2:
            # splitV = np.nanmean(np.array(mFMax), axis=0)
            splitV = np.nanmedian(np.array(mFMax), axis=0)
            for j in range(len(splitV)):
                if splitV[j] > mF[j]:
                    min = np.nanmin(np.array(mFMax)[:, j], axis=0)
                    mF[j] = [min, mF[j]][bool(min > mF[j])]
                else:
                    max = np.nanmax(np.array(mFMax)[:, j], axis=0)
                    mF[j] = [max, mF[j]][bool(max < mF[j])]
        mFMax.append(mF)
        # print(mFMax)
        mF = np.reshape(mF, (1, w))
    return mF


def f_sum(data):
    global fSum_pre
    # print(fSum_pre, data.shape)
    h, w = data.shape
    if h == 1:
        fSum_pre = data
        # print(fSum_pre)
        return data
    else:
        thred = np.full((w), 10000)
        s = np.vstack((fSum_pre, data[-1, :]))
        temp = np.nanmin(np.vstack((np.nansum(s, axis=0), thred)), axis=0)
        fSum_pre = temp
        return np.reshape(temp, (1, w))


def f_sum8h(data):
    h, w = data.shape
    s = np.zeros((1, w))
    s[0, :] = np.nan
    if h >= 8:
        s[0, :] = np.nansum(data[h - 8:h, :], axis=0)
    return s


def f_max(data):
    global fmax
    h, w = data.shape

    if h == 1:
        fmax = data
        return data

    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmax(np.vstack((fmax, data[-1, :])), axis=0)
    fmax = m
    return np.reshape(m, (1, w))


def f_min(data):
    global fmin
    h, w = data.shape

    if h == 1:
        fmin = data
        return data

    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmin(np.vstack((fmin, data[-1, :])), axis=0)
    fmin = m
    return np.reshape(m, (1, w))


def f_mean(data):
    h, w = data.shape
    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmean(data, axis=0)
    return np.reshape(m, (1, w))


def f_median(data):
    h, w = data.shape
    m = np.zeros(w)
    m[:] = np.nan
    m = np.nanmedian(data, axis=0)
    return np.reshape(m, (1, w))


def residual_value(feature):
    h, w = feature.shape
    data = np.full((h, w), x_mean[0:w])
    return feature - data


def cov_filter(feature):
    h, w = feature.shape
    if h < 3:
        return np.reshape(feature[-1, :], (1, w))
    f = np.nan_to_num(feature, True)
    result = np.full((1, w), np.nan)

    result = (f[-1, :] * 3 + f[-2, :] * 2 + f[-3, :]) / 6
    return np.reshape(result, (1, w))


def filter2d(feature, kernel):
    h, w = feature.shape
    kh, kw = kernel.shape
    if h < 5 or w < 5:
        return np.reshape(feature[-1, :], (1, w))
    r = np.sum(kernel)
    f = np.nan_to_num(feature, True)
    tmp = np.zeros((h + 2, w + 4), dtype='float64')
    tmp[0:-2, 2:-2] = f
    result = np.full((1, w), np.nan)
    for j in range(w):
        # print(tmp[-5:, j:j + kw])
        t = tmp[-5:, j:j + kw] * kernel
        result[0, j] = np.sum(t) / r
    return result


def compare(value, left, right):
    if np.isnan(value):
        return 0
    if value > right:
        return 1
    elif value > left:
        return 0
    else:
        return -1


def gen_obs(list):
    result = []
    result.append(compare(list[0], 60, 150))
    result.append(compare(list[1], 91, 99))
    result.append(compare(list[2], 36.5, 37.3))
    result.append(compare(list[3], 90, 139))
    result.append(compare(list[4], 65, 110))
    result.append(compare(list[5], 50, 92))
    result.append(compare(list[6], 12, 60))
    result.append(compare(list[7], 35, 45))
    result.append(compare(list[8], -10, 6))
    result.append(compare(list[9], 23, 30))
    result.append(compare(list[10], 0.6, 0.98))
    result.append(compare(list[11], 6, 8))
    return result


def get_obsfeature(feature):
    h, w = feature.shape
    result = []
    for i in range(h):
        result.append(gen_obs(feature[i, :12]))
    return np.reshape(result, (h, 12))


if __name__ == "__main__":
    # test = GetFeatures.readData('./training/p000050.psv')
    # feature = GetFeatures.getFeature(test)
    load_sepsis_model()

    # test = np.arange(1, 101, 1)
    # test = np.reshape(test, (50, 2))
    # result = []
    # print("src", test)
    # for t in range(50):
    #     current_data = test[:t + 1]
    #     if t == 0:
    #         result = genFeature(current_data)
    #     else:
    #         result = np.vstack((result, genFeature(current_data)))
    #
    # print("result", result.shape)
