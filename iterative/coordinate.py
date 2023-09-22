# Created by Churong Ji at 2021/6/7
import numpy as np
import math
import matplotlib.pyplot as plt
import ss2d


def remapping(img, alpha_degree, beta_degree, In):
    M, N = img.shape
    u0 = ((M + 1) / 2)
    v0 = ((N + 1) / 2)
    alpha = alpha_degree / 180 * math.pi
    beta = beta_degree / 180 * math.pi
    theta = 0

    Rx = np.array([[1, 0, 0], [0, math.cos(alpha), -math.sin(alpha)], [0, math.sin(alpha), math.cos(alpha)]])
    Ry = np.array([[math.cos(beta), 0, math.sin(beta)], [0, 1, 0], [-math.sin(beta), 0, math.cos(beta)]])
    Rz = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    R = np.matmul(Rz, np.matmul(Ry, Rx))
    H = np.matmul(In, np.matmul(np.linalg.inv(R), np.linalg.inv(In)))
    Pixel1 = np.matmul(H, np.array([1, 1, 1])) / np.dot(H[2], np.array([1, 1, 1]))
    Pixel2 = np.matmul(H, np.array([1, N, 1])) / np.dot(H[2], np.array([1, N, 1]))
    Pixel3 = np.matmul(H, np.array([M, 1, 1])) / np.dot(H[2], np.array([M, 1, 1]))
    Pixel4 = np.matmul(H, np.array([M, N, 1])) / np.dot(H[2], np.array([M, N, 1]))
    O2 = np.matmul(H, np.array([u0, v0, 1])) / np.dot(H[2], np.array([u0, v0, 1]))
    height = round(max([Pixel1[0], Pixel2[0], Pixel3[0], Pixel4[0]])
                   - min([Pixel1[0], Pixel2[0], Pixel3[0], Pixel4[0]]))
    width = round(max([Pixel1[1], Pixel2[1], Pixel3[1], Pixel4[1]])
                  - min([Pixel1[1], Pixel2[1], Pixel3[1], Pixel4[1]]))

    # compute img_new
    img_new = np.zeros((height, width))
    delta_u = round(min([Pixel1[0], Pixel2[0], Pixel3[0], Pixel4[0]]))
    delta_v = round(min([Pixel1[1], Pixel2[1], Pixel3[1], Pixel4[1]]))
    u00 = O2[0] - delta_u  # 主点在新图中的像素坐标[u00,v00],注意保留小数
    v00 = O2[1] - delta_v
    H_inv = np.linalg.inv(H)
    for u2 in range(delta_u + 1, delta_u + height + 1):
        for v2 in range(delta_v + 1, delta_v + width + 1):
            temp = np.matmul(H_inv, np.array([u2, v2, 1]))
            pix = temp / temp[2]
            if 1 <= pix[0] <= M and 1 <= pix[1] <= N:
                u1 = pix[0]
                v1 = pix[1]
                u11 = int(u1)  # same as math.floor for positive numbers
                u12 = u11 + 1
                v11 = int(v1)
                v12 = v11 + 1
                img_new[u2 - delta_u - 1][v2 - delta_v - 1] = (u12 - u1) * (v12 - v1) * img[u11 - 1][v11 - 1] + \
                                                              (u12 - u1) * (v1 - v11) * img[u11 - 1][v12 - 1] + \
                                                              (u1 - u11) * (v12 - v1) * img[u12 - 1][v11 - 1] + \
                                                              (u1 - u11) * (v1 - v11) * img[u12 - 1][v12 - 1]
    return img_new, u00, v00, delta_u, delta_v, H


def pixel_coordinate(u2, v2, H_inv, delta_u, delta_v):
    u2 = u2 + delta_u
    v2 = v2 + delta_v
    temp = np.matmul(H_inv, np.array([u2, v2, 1]))
    pix = temp / temp[2]
    return pix[0], pix[1]


def helper_world(phase001, phase002, T):
    phase01 = np.unwrap(phase001)
    phase02 = np.unwrap(phase002)
    line01 = (phase01[-1] - phase01[0]) / (2 * math.pi) * T
    line02 = (phase02[-1] - phase02[0]) / (2 * math.pi) * T
    return line01, line02


def world_coordinate(I, T):
    # returns 3x1 np array (reshape from vector)
    M1, N1 = I.shape
    uO3 = math.ceil((M1 + 1) / 2)
    vO3 = math.ceil((N1 + 1) / 2)
    up = M1 / 4
    vp = N1 / 4

    # 选择局部子图，使用phase estimation提取物平面特征点A，B，C，D的相位
    row_num, col_num = 181, 181
    row_start = [up - (row_num - 1) / 2, uO3 - (row_num - 1) / 2]
    row_end = [up + (row_num - 1) / 2, uO3 + (row_num - 1) / 2]
    col_start = [vp - (col_num - 1) / 2, vO3 - (col_num - 1) / 2]
    col_end = [vp + (col_num - 1) / 2, vO3 + (col_num - 1) / 2]

    # 相位估计法计算特征点的世界坐标
    phase001, phase002 = ss2d.length1(M1 - up, I, row_num, row_start[0], row_end[0], col_start[1], col_end[1])
    line01, line02 = helper_world(phase001, phase002, T)
    Aw = np.reshape(np.array([-line01, -line02, 0]), (3, 1))
    phase001, phase002 = ss2d.length1(M1 + up, I, row_num, row_start[1], row_end[1], col_start[1], col_end[1])
    line01, line02 = helper_world(phase001, phase002, T)
    Cw = np.reshape(np.array([line01, line02, 0]), (3, 1))
    phase001, phase002 = ss2d.length3(N1 - vp, I, col_num, row_start[1], row_end[1], col_start[0], col_end[0])
    line01, line02 = helper_world(phase001, phase002, T)
    Bw = np.reshape(np.array([-line01, -line02, 0]), (3, 1))
    phase001, phase002 = ss2d.length3(N1 + vp, I, col_num, row_start[1], row_end[1], col_start[1], col_end[1])
    line01, line02 = helper_world(phase001, phase002, T)
    Dw = np.reshape(np.array([line01, line02, 0]), (3, 1))
    return Aw, Bw, Cw, Dw


def coordinate(img, alpha_degree, beta_degree, T, f, psize):
    # camera model
    M, N = img.shape
    u0 = ((M + 1) / 2)
    v0 = ((N + 1) / 2)
    dX = psize
    dY = psize
    alpha_x = f / dX
    alpha_y = f / dY
    gamma = 0
    In = np.array([[alpha_x, gamma, u0], [0, alpha_y, v0], [0, 0, 1]])  # Intrinsic matrix

    # pixel coordinate (u,v,1)
    img_new, u00, v00, delta_u, delta_v, H = remapping(img, alpha_degree, beta_degree, In)
    # 截取校正后图像的中点对称部分[400 * 400]
    I = img_new[round(u00 - 200.5):round(u00 + 199.5), round(v00 - 200.5):round(v00 + 199.5)]
    M1, N1 = I.shape
    u000 = math.ceil((M1 + 1) / 2)
    v000 = math.ceil((N1 + 1) / 2)
    uO3 = u000
    vO3 = v000
    # u3, v3 is the coordinate array for point A, B, C, D in I
    up = M1 / 4
    vp = N1 / 4
    u3 = np.array([up, uO3, M1 - up, uO3])  # corresponds to A, B, C, D respectively
    v3 = np.array([vO3, vp, vO3, N1 - vp])
    # u2, v2 is the coordinate array for point A, B, C, D in the complete img_new
    shift_u = u00 - u000
    shift_v = v00 - v000
    u2 = u3 + shift_u
    v2 = v3 + shift_v
    # u1, v1 is the pixel coordinate array for point A, B, C, D
    u1 = np.zeros(4)
    v1 = np.zeros(4)
    H_inv = np.linalg.inv(H)
    for i in range(4):
        u1[i], v1[i] = pixel_coordinate(u2[i], v2[i], H_inv, delta_u, delta_v)

    # camera coordinate (Xc, Yc, Zc) & Zc = 1
    inverse_In = np.linalg.inv(In)
    n = u1.shape[0]
    temp = np.zeros((n, 3))  # nx3 matrix where n is the number of control points
    for i in range(n):
        Pix_i = np.array([u1[i], v1[i], 1])
        temp[i] = np.matmul(inverse_In, Pix_i)
    temp = np.transpose(temp)  # now a 3xn matrix
    Qc = temp[:-1, :]  # 2xn matrix, no depth info now
    # plot 4 control points in image plane
    # plt.figure()
    # plt.axis([-0.5, 0.5, -0.5, 0.5])
    # plt.plot(Qc[0], Qc[1])
    # plt.plot(Qc[0], Qc[1], 'ro')
    # for i_x, i_y in zip(np.round(Qc[0], decimals=4), np.round(Qc[1], decimals=4)):
    #     plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
    # plt.title('Control point on normalized image plane')
    # plt.xlabel('Position Xc/um')
    # plt.ylabel('Position Yc/um')
    # plt.show()

    # world coordinate (Xw,Yw,Zw=0)
    Aw, Bw, Cw, Dw = world_coordinate(I, T)
    Pw = np.hstack((Aw, Bw, Cw, Dw))  # 3xn world coordinate matrix
    # Pw_line = np.hstack((Aw, Bw, Cw, Dw, Aw))
    # plt.figure()
    # # plt.axis([-2000, 2000, -2000, 2000])
    # plt.plot(Pw[0], Pw[1], 'bo')
    # plt.plot(Pw_line[0], Pw_line[1], 'r')
    # for i_x, i_y in zip(np.round(Pw[0], decimals=4), np.round(Pw[1], decimals=4)):
    #     plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
    # plt.title('Control point on object plane')
    # plt.xlabel('Position Xw/um')
    # plt.ylabel('Position Yw/um')
    # plt.show()
    return Qc, Pw, I
