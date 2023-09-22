# Created by Churong Ji at 2021/6/7
import multiprocessing

import numpy as np
import math
from math import cos, sin, pi
from scipy.optimize import least_squares
import ss2d



def window2(N, M):
    """ Use Hanning window."""
    wc = np.hanning(N)
    wr = np.hanning(M)
    maskr, maskc = np.meshgrid(wr, wc)
    return maskr * maskc


def find_max_amplitude(S_value, midpoint):
    """ Helper function for SS2D."""
    result = np.where(S_value == np.max(S_value))
    if len(result[0]) == 1:
        tmp_S = np.copy(S_value)
        row1 = result[0][0]
        col1 = result[1][0]
        tmp_S[row1][col1] = 0
        result = np.where(tmp_S == np.max(tmp_S))
        row2 = result[0][0]
        col2 = result[1][0]
        if col1 > col2:
            row = np.array([row2, row1])
            col = np.array([col2, col1])
        elif col1 == col2:
            if row1 > row2:
                row = np.array([row2, row1])
            else:
                row = np.array([row1, row2])
            col = np.array([col1, col2])
        else:
            row = np.array([row1, row2])
            col = np.array([col1, col2])
    else:
        if result[1][0] > result[1][1]:  # sort by column as matlab does
            row = np.flip(result[0])
            col = np.flip(result[1])
        else:
            row = result[0]
            col = result[1]
    u = row[1] - midpoint[0] + 1
    v = col[1] - midpoint[1] + 1
    return u, v, row + 1, col + 1


def SS2D(data):
    """
    计算 2D pattern中心点相位 (PhaseX,PhaseY),实现phase estimation method
    对于 SS2D.m的优化是：确定了X，Y的方位，即始终选取第一组点在第一四象限
    """
    data = data - np.average(data)  # 去data直流分量
    N, M = data.shape
    midpoint = [math.ceil((N + 1) / 2), math.ceil((M + 1) / 2)]
    # 中心点坐标[*, *] % ceil返回大于或者等于指定表达式的最小整数
    win = window2(N, M)
    # TODO: optimize fft -- only calculate one point
    Spectrum = np.fft.fftshift(np.fft.fft2(data * win))
    S_value0 = abs(Spectrum)
    S_value1 = np.copy(S_value0)
    u1, v1, row1, col1 = find_max_amplitude(S_value1, midpoint)
    # 为了使选取的第一组点始终是在一四象限
    if row1[1] > midpoint[0]:
        point1 = 2
        k1 = ((u1 + 1) / N) / ((v1 + 1) / M)
        if k1 == 0:
            theta1 = 90
        else:
            theta1 = math.atan(1 / k1) / pi * 180  # unit: degree
    else:
        S_value1[row1[0] - 3: row1[0] + 1, col1[0] - 3: col1[0] + 1] = 0
        S_value1[row1[1] - 3: row1[1] + 1, col1[1] - 3: col1[1] + 1] = 0
        u1, v1, row1, col1 = find_max_amplitude(S_value1, midpoint)
        point1 = 2
        if v1 == 0:
            k1 = np.double('inf')
        else:
            k1 = (u1 / N) / (v1 / M)
        if k1 == 0:
            theta1 = 90
        else:
            theta1 = math.atan(1 / k1) / pi * 180  # unit: degree

    # S_value2 找二维频域上另一对幅值最大点 C,D
    S_value2 = np.copy(S_value0)
    S_value2[row1[0] - 3: row1[0] + 3, col1[0] - 3: col1[0] + 3] = 0
    S_value2[row1[1] - 3: row1[1] + 3, col1[1] - 3: col1[1] + 3] = 0
    u2, v2, row2, col2 = find_max_amplitude(S_value2, midpoint)
    point2 = 2
    if u2 > 0:
        u2 = row2[0] - midpoint[0]
        v2 = col2[0] - midpoint[1]
        point2 = 1
    if v2 == 0:
        k2 = np.double('inf')
    else:
        k2 = (u2 / N) / (v2 / M)
    if k2 == 0:
        theta2 = 90
    else:
        theta2 = math.atan(1 / k2) / pi * 180
    it = 0
    while (abs(theta1 - theta2) < 80) | (abs(theta1 - theta2) > 100):
        it += 1
        temp11 = max(row2[0] - 2, 1)
        temp12 = min(row2[0] + 2, N)
        temp13 = max(col2[0] - 2, 1)
        temp14 = min(col2[0] + 2, M)
        temp21 = max(row2[1] - 2, 1)
        temp22 = min(row2[1] + 2, N)
        temp23 = max(col2[1] - 2, 1)
        temp24 = min(col2[1] + 2, M)
        S_value2[temp11 - 1: temp12 + 1, temp13 - 1: temp14 + 1] = 0
        S_value2[temp21 - 1: temp22 + 1, temp23 - 1: temp24 + 1] = 0
        u2, v2, row2, col2 = find_max_amplitude(S_value2, midpoint)
        point2 = 2
        if u2 > 0:
            u2 = row2[0] - midpoint[0]
            v2 = col2[0] - midpoint[1]
            point2 = 1
        if v2 == 0:
            k2 = np.double('inf')
        else:
            k2 = (u2 / N) / (v2 / M)
        if k2 == 0:
            theta2 = 90
        else:
            theta2 = math.atan(1 / k2) / pi * 180

    # 计算得到图像中心点C沿X和Y方向的相位phase1, phase2
    phase1 = np.angle(Spectrum[row1[point1 - 1] - 1][col1[point1 - 1] - 1]) \
             + pi * u1 * (N - 1) / N + pi * v1 * (M - 1) / M
    phase1 = phase1 % (2 * pi)
    phase2 = np.angle(Spectrum[row2[point2 - 1] - 1][col2[point2 - 1] - 1]) \
             + pi * u2 * (N - 1) / N + pi * v2 * (M - 1) / M
    phase2 = phase2 % (2 * pi)

    # 计算pattern在XY平面内的转角theta
    if theta1 < 0:
        theta = 180 + theta1
    else:
        theta = theta1
    return phase1, phase2, u1, v1, u2, v2, row1, col1, row2, col2, theta, Spectrum


def SS3D(data, T, p_num):
    """Calculate in-plane 3DOF: tx, ty, theta."""
    N, M = data.shape
    (v0, u0) = (math.ceil((N + 1) / 2), math.ceil((M + 1) / 2))

    # calculate theta
    w_half = int(p_num / 2)
    w_end = M - p_num + 1
    phase1 = np.zeros(w_end)
    phase2 = np.zeros(w_end)
    for k in range(w_end):
        I1 = data[v0 - w_half - 1:v0 + w_half, k:k + p_num]
        phase1[k], phase2[k], _, _, _, _, _, _, _, _, _, _ = SS2D(I1)
    phase1 = np.unwrap(phase1)
    phase2 = np.unwrap(phase2)
    delta1 = abs(phase1[w_end - 1] - phase1[0])
    delta2 = abs(phase2[w_end - 1] - phase2[0])
    theta_hat = abs(math.atan(delta1 / delta2))

    # calculate tx, ty
    _, _, _, _, _, _, row_1, col_1, row_2, col_2, _, spectrum = SS2D(data)
    row1, col1, row2, col2 = row_1[1], col_1[1], row_2[1], col_2[1]
    temp1 = np.angle(spectrum[row1 - 1][col1 - 1]) + pi * (row1 - v0) * (N - 1) / N + pi * (col1 - u0) * (M - 1) / M
    temp2 = np.angle(spectrum[row2 - 1][col2 - 1]) + pi * (row2 - v0) * (N - 1) / N + pi * (col2 - u0) * (M - 1) / M
    phase_1 = np.unwrap(np.array([temp1]))[0]
    phase_2 = np.unwrap(np.array([temp2]))[0]
    tx_hat = (phase_1 % (2 * pi)) * T / (2 * pi)
    ty_hat = (phase_2 % (2 * pi)) * T / (2 * pi)
    return tx_hat, ty_hat, theta_hat


def phase_to_length(phase001, phase002, term1, term2, T):
    """Helper function of SS6D."""
    phase01 = np.unwrap(phase001)
    phase02 = np.unwrap(phase002)
    line01 = (phase01[int(term1 - 2 * term2)] - phase01[0]) / (2 * pi) * T
    line02 = (phase02[int(term1 - 2 * term2)] - phase02[0]) / (2 * pi) * T
    return math.sqrt(line01 * line01 + line02 * line02)


def SS6D(data, T, f, p_size, p_num):
    """在弱透视模型下, 基于几何关系 geometry-based pose estimation, 估计pattern的6DOF的位移和角度."""
    # parameters
    N, M = data.shape
    p = f / p_size
    (u0, v0) = (math.ceil((M + 1) / 2), math.ceil((N + 1) / 2))
    const_1 = p / v0
    const_2 = p / u0

    # calculate in-plane tx, ty, theta
    tx_hat, ty_hat, theta_hat = SS3D(data, T, p_num)
    # tx_hat, ty_hat, theta_hat = 0., 0., 0.

    # 弱透视近似模型，估计 alpha、beta、tz
    # define the coordinate of point A, B, C, D in the image plane
    (up, vp) = (M / 4, N / 4)
    (uA, uB, uC, uD) = (M - up, M - up, up, up)
    (vA, vB, vC, vD) = (N - vp, vp, vp, N - vp)
    # 选择局部子图，使用phase estimation提取物平面特征点A，B，C，D的相位
    row_num, col_num = 201, 201
    row_start = vp - (row_num - 1) / 2
    row_end = vp + (row_num - 1) / 2
    col_start = up - (col_num - 1) / 2
    col_end = up + (col_num - 1) / 2

    # 计算4个边长：CD,AB,BC,DA
    phase001, phase002 = ss2d.length1(N, data, row_num, row_start, row_end, col_start, col_end)
    length_CD = phase_to_length(phase001, phase002, N, vp, T)
    phase001, phase002 = ss2d.length2(N, data, row_num, row_start, row_end, col_start, col_end, M)
    length_AB = phase_to_length(phase001, phase002, N, vp, T)
    phase001, phase002 = ss2d.length3(M, data, col_num, row_start, row_end, col_start, col_end)
    length_BC = phase_to_length(phase001, phase002, M, up, T)
    phase001, phase002 = ss2d.length4(M, data, col_num, row_start, row_end, col_start, col_end, N)
    length_DA = phase_to_length(phase001, phase002, M, up, T)

    # 确定3个变量的搜索起始点
    tz1 = length_AB * const_1
    tz2 = length_BC * const_2
    tz3 = length_CD * const_1
    tz4 = length_DA * const_2
    tz0 = (tz1 + tz2 + tz3 + tz4) / 4
    alpha0 = math.asin(2 * (tz2 - tz4) / (length_AB + length_CD))
    beta0 = math.asin(2 * (tz1 - tz3) / (length_BC + length_DA))

    def func(x):
        function = (((uA - u0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                (uA - u0) * sin(x[1] * beta0) - (vA - v0) * sin(x[0] * alpha0) * cos(x[1] * beta0) - p * cos(
            x[1] * beta0) * cos(x[0] * alpha0)) -
                     (uB - u0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uB - u0) * sin(x[1] * beta0) - (vB - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 +
                    ((vA - v0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                            (uA - u0) * sin(x[1] * beta0) - (vA - v0) * sin(x[0] * alpha0) * cos(
                        x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (vB - v0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uB - u0) * sin(x[1] * beta0) - (vB - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 +
                    ((-p) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                            (uA - u0) * sin(x[1] * beta0) - (vA - v0) * sin(x[0] * alpha0) * cos(
                        x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (-p) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uB - u0) * sin(x[1] * beta0) - (vB - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 - length_AB ** 2) ** 2 + \
                   (((uB - u0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                           (uB - u0) * sin(x[1] * beta0) - (vB - v0) * sin(x[0] * alpha0) * cos(
                       x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (uC - u0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uC - u0) * sin(x[1] * beta0) - (vC - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 +
                    ((vB - v0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                            (uB - u0) * sin(x[1] * beta0) - (vB - v0) * sin(x[0] * alpha0) * cos(
                        x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (vC - v0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uC - u0) * sin(x[1] * beta0) - (vC - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 +
                    ((-p) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                            (uB - u0) * sin(x[1] * beta0) - (vB - v0) * sin(x[0] * alpha0) * cos(
                        x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (-p) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uC - u0) * sin(x[1] * beta0) - (vC - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 - length_BC ** 2) ** 2 + \
                   (((uC - u0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                           (uC - u0) * sin(x[1] * beta0) - (vC - v0) * sin(x[0] * alpha0) * cos(
                       x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (uD - u0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uD - u0) * sin(x[1] * beta0) - (vD - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 +
                    ((vC - v0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                            (uC - u0) * sin(x[1] * beta0) - (vC - v0) * sin(x[0] * alpha0) * cos(
                        x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (vD - v0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uD - u0) * sin(x[1] * beta0) - (vD - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 +
                    ((-p) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                            (uC - u0) * sin(x[1] * beta0) - (vC - v0) * sin(x[0] * alpha0) * cos(
                        x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (-p) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uD - u0) * sin(x[1] * beta0) - (vD - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 - length_CD ** 2) ** 2 + \
                   (((uD - u0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                           (uD - u0) * sin(x[1] * beta0) - (vD - v0) * sin(x[0] * alpha0) * cos(
                       x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (uA - u0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uA - u0) * sin(x[1] * beta0) - (vA - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 +
                    ((vD - v0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                            (uD - u0) * sin(x[1] * beta0) - (vD - v0) * sin(x[0] * alpha0) * cos(
                        x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (vA - v0) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uA - u0) * sin(x[1] * beta0) - (vA - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 +
                    ((-p) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                            (uD - u0) * sin(x[1] * beta0) - (vD - v0) * sin(x[0] * alpha0) * cos(
                        x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0)) -
                     (-p) * (x[2] * tz0) * cos(x[0] * alpha0) * cos(x[1] * beta0) / (
                             (uA - u0) * sin(x[1] * beta0) - (vA - v0) * sin(x[0] * alpha0) * cos(
                         x[1] * beta0) - p * cos(x[1] * beta0) * cos(x[0] * alpha0))) ** 2 - length_DA ** 2) ** 2
        return function
    result = least_squares(func, x0=np.array([1, 1, 1]), args=())
    x = result['x']
    alpha_hat = x[0] * alpha0
    beta_hat = x[1] * beta0
    tz_hat = x[2] * tz0
    return tx_hat, ty_hat, tz_hat, alpha_hat, beta_hat, theta_hat
