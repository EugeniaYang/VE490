# Created by Churong Ji at 2021/6/6
import math
from math import pi
import numpy as np
import cqr


def calculate_length(data, range_li, data_range_li, flag):
    """Helper function.
    flag is True when called by length1 and length2, false otherwise."""
    i = 0
    phase1, phase2 = [], []
    prev_row1, prev_col1, prev_row2, prev_col2 = 0, 0, 0, 0
    for k in range(range_li[0], range_li[1]):
        i += 1
        L_square = 10  # 搜索框单侧边长.unit:pixel
        if flag:
            data_im = np.copy(
                data[k + data_range_li[0]:k + data_range_li[1] + 1, data_range_li[2]:data_range_li[3] + 1])
        else:
            data_im = np.copy(
                data[data_range_li[0]:data_range_li[1] + 1, k + data_range_li[2]:k + data_range_li[3] + 1])
        N, M = data_im.shape
        midpoint = (math.ceil((N + 1) / 2), math.ceil((M + 1) / 2))
        win = cqr.window2(N, M)

        if i != 1:  # 计算初始位置其余若干张子图片的phase
            # TODO: optimize fft -- only calculate one point
            spectrum = np.fft.fftshift(np.fft.fft2(data_im * win))
            # fft = pyfftw.builders.fft2(data_im * win, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
            # tmp = fft()
            # spectrum = pyfftw.interfaces.scipy_fftpack.fftshift(tmp)
            S_value = abs(spectrum)
            # Search Square list in the order of top, bottom, left, right
            SSquare_1 = [max(prev_row1 - L_square, 1), min(prev_row1 + L_square, N),
                         max(prev_col1 - L_square, 1), min(prev_col1 + L_square, M)]
            SSquare_2 = [max(prev_row2 - L_square, 1), min(prev_row2 + L_square, N),
                         max(prev_col2 - L_square, 1), min(prev_col2 + L_square, M)]
            search_mat_1_D1 = S_value[SSquare_1[0] - 1:SSquare_1[1], SSquare_1[2] - 1:SSquare_1[3]]
            search_mat_1_D2 = S_value[SSquare_2[0] - 1:SSquare_2[1], SSquare_2[2] - 1:SSquare_2[3]]

            # 找第一个方向的下一点基频
            result = np.where(search_mat_1_D1 == np.max(search_mat_1_D1))
            (row1, col1) = (result[0][0] + SSquare_1[0], result[1][0] + SSquare_1[2])
            (u1, v1) = (row1 - midpoint[0], col1 - midpoint[0])
            # 找第二个方向的下一点基频
            result = np.where(search_mat_1_D2 == np.max(search_mat_1_D2))
            (row2, col2) = (result[0][0] + SSquare_2[0], result[1][0] + SSquare_2[2])
            (u2, v2) = (row2 - midpoint[0], col2 - midpoint[1])
            if u2 > 0:
                (u2, v2) = (row2 - midpoint[0], col2 - midpoint[1])
            prev_row1, prev_col1, prev_row2, prev_col2 = row1, col1, row2, col2
            phase01 = np.angle(spectrum[row1 - 1][col1 - 1]) + pi * u1 * (N - 1) / N + pi * v1 * (M - 1) / M
            phase02 = np.angle(spectrum[row2 - 1][col2 - 1]) + pi * u2 * (N - 1) / N + pi * v2 * (M - 1) / M
            phase1.append(phase01 % (2 * pi))
            phase2.append(phase02 % (2 * pi))

        else:  # 计算初始位置的第一张图片的phase
            phase01, phase02, u1, v1, u2, v2, _, _, _, _, _, _ = cqr.SS2D(data_im)
            phase1.append(phase01)
            phase2.append(phase02)
            prev_row1 = u1 + midpoint[0]
            prev_col1 = v1 + midpoint[1]
            prev_row2 = u2 + midpoint[0]
            prev_col2 = v2 + midpoint[1]

    return np.array(phase1), np.array(phase2)


def length1(M_in, data, row_num, row_start, row_end, col_start, col_end):
    """计算image短边的两个控制点CD之间的距离."""
    range_li = [int(row_start), int(M_in - row_end + 2)]
    data_range_li = [-1, int(row_num - 2), int(col_start - 1), int(col_end - 1)]
    phase1, phase2 = calculate_length(data, range_li, data_range_li, True)
    return phase1, phase2


def length2(M_in, data, row_num, row_start, row_end, col_start, col_end, N_in):
    """计算image短边的两个控制点AB之间的距离."""
    range_li = [int(row_start), int(M_in - row_end + 2)]
    data_range_li = [-1, int(row_num - 2), int(N_in - col_end), int(N_in - col_start)]
    phase1, phase2 = calculate_length(data, range_li, data_range_li, True)
    return phase1, phase2


def length3(N_in, data, col_num, row_start, row_end, col_start, col_end):
    """计算image水平方向的两个控制点BC之间的距离."""
    range_li = [int(col_start), int(N_in - col_end + 2)]
    data_range_li = [int(row_start - 1), int(row_end - 1), -1, int(col_num - 2)]
    phase1, phase2 = calculate_length(data, range_li, data_range_li, False)
    return phase1, phase2


def length4(N_in, data, col_num, row_start, row_end, col_start, col_end, M_in):
    """计算image长边的两个控制点BC之间的距离."""
    range_li = [int(col_start), int(N_in - col_end + 2)]
    data_range_li = [int(M_in - row_end), int(M_in - row_start), -1, int(col_num - 2)]
    phase1, phase2 = calculate_length(data, range_li, data_range_li, False)
    return phase1, phase2
