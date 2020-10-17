import numpy as np
from scipy.interpolate import interp1d


def get_diff(data, used_cols=None, diff_step=1):
    '''
    Calculates the difference of each feature in a sample over diff_step time steps (approx. first derivative)
    while IGNORING NaNs

    :param data: The data for which to calculate the differences
    :param used_cols: Which features to calculate the differences for (array of indices)
    :param diff_step: over how many time steps the difference will be calculated
    :return: the difference data (has NaNs where the input data had NaNs)
                The first diff_step non-NaN values of each feature are 0
    '''
    dims = np.shape(data)
    if len(dims) < 2:
        dims = dims[:, np.newaxis]

    if used_cols is None:
        used_cols = list(range(dims[1]))

    out = np.zeros((dims[0], len(used_cols)))
    for c in range(len(used_cols)):
        n = used_cols[c]
        col = data[:, n]

        col_noNaNs = col[np.isnan(col) == False]  # delete NaNs
        diff_noNaNs = np.zeros(len(col_noNaNs))
        # get difference in no - NaN space
        diff_noNaNs[diff_step:] = col_noNaNs[diff_step:] - col_noNaNs[: -diff_step]

        out[np.isnan(col) == False, c] = diff_noNaNs  # insert values where there are no NaNs

    out[np.isnan(data[:, used_cols])] = np.nan
    return out


def _over24(col_in):
    le = int(np.ceil(len(col_in) / 24) * 24)
    col = np.zeros(le)
    col[:len(col_in)] = col_in[:, 0]
    col = np.reshape(col, (24, -1))
    col_max = np.max(col, 0)
    col = np.repeat(col_max[np.newaxis, :], 24, 0)
    col = col.flatten()
    col = col[:len(col_in)]

    diff = np.append([0], col_max[1:] - col_max[:-1])
    diff = np.repeat(diff[np.newaxis, :], 24, 0)
    diff = diff.flatten()
    diff = diff[:len(col_in)]

    return col[:, np.newaxis], diff[:, np.newaxis]


def get_qSOFA(data_in):
    resp = data_in[:, 6]
    resp_high = (resp >= 22).astype(int)

    sbp = data_in[:, 3]
    sbp_low = (sbp <= 100).astype(int)

    return np.stack([resp_high, sbp_low], 1), ["resp_high_qsofa", "sbp_low_qsofa"]


def get_SOFA(data_in):
    '''
    Calculates the SOFA score and all its components
    as well as the difference over the last 24h window (on both SOFA and its components).
    -> Every 24h window has the same score (i.e. the worst score)

    :param data_in:
    :return:
    '''
    # Breathing
    O2Sat = data_in[:, 1]
    O2Sat[O2Sat < 0] = 0
    x = [0, 32, 50, 67, 75, 84, 90, 95, 98, 100]
    y = [2, 20, 28, 35, 40, 50, 60, 70, 80, 100]
    PaO2 = interp1d(x, y, kind='linear')(O2Sat)
    FiO2 = data_in[:, 10]
    FiO2[FiO2 <= 0] = 0.001
    PaFi = PaO2 / FiO2
    PaFi = np.concatenate([PaFi[:, np.newaxis], 500 * np.ones((len(data_in[:, 0]), 1))], 1)
    PaFi = np.nanmin(PaFi, 1)
    PaFi_points = interp1d([0, 100, 200, 300, 400, 100000], [4, 3, 2, 1, 0, 0], kind='previous')(PaFi)
    PaFi_points[np.isnan(PaFi_points)] = 0
    PaFi_points = PaFi_points[:, np.newaxis]
    PaFi_worst, PaFi_diff = _over24(PaFi_points)

    # Cardiovascular
    MAP = data_in[:, 4]
    MAP[MAP < 0] = 0
    MAP_points = interp1d([0, 70, 1000], [1, 0, 0], kind='previous')(MAP)
    MAP_points[np.isnan(MAP_points)] = 0;
    MAP_points = MAP_points[:, np.newaxis]
    MAP_worst, MAP_diff = _over24(MAP_points);

    # Liver
    Liver = data_in[:, 20]
    Liver[Liver < 0] = 0
    Liver_points = interp1d([0, 1.2, 2, 6, 12, 1000], [0, 1, 2, 3, 4, 4], kind='previous')(Liver)
    Liver_points[np.isnan(Liver_points)] = 0
    Liver_points = Liver_points[:, np.newaxis]
    [Liver_worst, Liver_diff] = _over24(Liver_points)

    # Platelets
    Platelets = data_in[:, 33]
    Platelets[Platelets < 0] = 0
    Platelet_points = interp1d([0, 20, 50, 100, 150, 10000], [4, 3, 2, 1, 0, 0], kind='previous')(Platelets)
    Platelet_points[np.isnan(Platelet_points)] = 0
    Platelet_points = Platelet_points[:, np.newaxis]
    [Platelet_worst, Platelet_diff] = _over24(Platelet_points)

    # Kidneys
    Kidney = data_in[:, 19]
    Kidney[Kidney < 0] = 0
    Kidney_points = interp1d([0, 1.2, 2, 3.5, 5, 1000], [0, 1, 2, 3, 4, 4], kind='previous')(Kidney)
    Kidney_points[np.isnan(Kidney_points)] = 0
    Kidney_points = Kidney_points[:, np.newaxis]
    [Kidney_worst, Kidney_diff] = _over24(Kidney_points)

    # calculate approximated SOFA score
    SOFA = PaFi_worst + MAP_worst + Liver_worst + Platelet_worst + Kidney_worst
    [SOFA_24, SOFA_diff] = _over24(SOFA)

    # used_features
    data_out = [PaFi_points, PaFi_worst, PaFi_diff, MAP_points, MAP_worst, MAP_diff,
                Liver_points, Liver_worst, Liver_diff, Platelet_points, Platelet_worst, Platelet_diff,
                Kidney_points, Kidney_worst, Kidney_diff, SOFA_24, SOFA_diff]

    labels = ["PaFi_points", "PaFi_worst", "PaFi_diff", "MAP_points", "MAP_worst", "MAP_diff",
              "Liver_points", "Liver_worst", "Liver_diff", "Platelet_points", "Platelet_worst", "Platelet_diff",
              "Kidney_points", "Kidney_worst", "Kidney_diff", "SOFA_24", "SOFA_diff"]

    return np.concatenate(data_out, 1), labels


def last_reliable(data_in):
    dims = np.shape(data_in)
    out = np.zeros(dims)

    non_nans = np.invert(np.isnan(data_in))
    counter = np.zeros(dims[1])

    for row in range(dims[0]):
        counter = counter + 1
        counter[non_nans[row, :]] = 0

        out[row, :] = counter
    return out
