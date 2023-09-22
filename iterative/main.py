import pathlib
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import cqr
import scipy.sparse.linalg as sla
import time
import coordinate
import math
import imageio
import decode


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def xform(P, R, t):
    """Transform the 3D point set P by rotation R and translation t."""
    return np.matmul(R, P) + t


def xformproj(P, R, t):
    """Transform the 3D point set P by rotation R and translation t,
        and then project them to the normalized image plane."""
    Q = xform(P, R, t)
    return Q[0:2, :] / Q[2, :]


def qmatQ(q):
    """ Compute the Q matrix (4x4) of quaternion q."""
    [w, x, y, z] = q
    return np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])


def qmatW(q):
    """Compute the W matrix (4x4) of quaternion q."""
    [w, x, y, z] = q
    return np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])


def quat2mat(q):
    """Convert a quaternion to a 3x3 rotation matrix."""
    [a, b, c, d] = q
    return np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a ** 2 + c ** 2 - b ** 2 - d ** 2, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a ** 2 + d ** 2 - b ** 2 - c ** 2]])


def abskernel(P, Q, F, G, method):
    """Solve the intermediate absolute orientation problems
        in the inner loop of the OI pose estimation algorithm."""
    n = P.shape[1]
    Q = Q.T
    for i in range(n):
        Q[i] = np.matmul(F[i], Q[i])
    Q = Q.T  # 3xn matrix now
    pbar = np.sum(P, axis=1) / n
    qbar = np.sum(Q, axis=1) / n
    P = P - np.reshape(pbar, (3, 1))
    Q = Q - np.reshape(qbar, (3, 1))
    R = np.zeros((3, 3))

    if method == 'SVD':
        M = np.zeros((3, 3))
        for i in range(n):
            M += np.matmul(np.reshape(P[0:4, i], (3, 1)), np.reshape(Q.T[i], (1, 3)))
        U, S, V = np.linalg.svd(M)
        R = np.matmul(V.T, U.T)
    elif method == 'QTN':
        A = np.zeros((4, 4))
        for i in range(n):
            A += np.matmul(qmatQ(np.hstack(([1], Q.T[i]))).T, qmatW(np.hstack(([1], P.T[i]))))
        V, D = sla.eigs(A, k=1, which='LM', tol=0, M=np.identity(4))
        R = quat2mat(V)
    else:
        print("Method not defined!")
        exit(1)

    sumgh = np.zeros((3, 1))
    for i in range(n):
        sumgh += np.reshape(np.matmul(F[i], np.matmul(R, P.T[i])), (3, 1))
    t = np.matmul(G, sumgh)
    Qout = xform(P, R, t)

    # calculate error
    err2 = 0
    for i in range(n):
        vec = np.matmul(np.identity(3) - F[i], Qout.T[i])
        err2 += np.dot(vec, vec)
    return R, t, Qout, err2


def objpose(P, Qp, options, nargout):
    ghflag = 0
    n = P.shape[1]
    pbar = np.sum(P, axis=1) / n  # sum rows
    P = np.transpose(P.T - pbar)
    Q = np.vstack((Qp, np.ones((n,))))
    # compute projection matrices
    F = np.zeros((n, 3, 3))
    for i in range(n):
        V = np.reshape(Q.T[i] / Q[2][i], (3, 1))
        F[i] = np.matmul(V, V.T) / np.matmul(V.T, V)

    # compute the matrix factor required to compute t
    t_factor = np.linalg.inv(np.identity(3) - np.sum(F, axis=0) / n) / n
    it = 0
    if 'initR' in options:  # initial guess of rotation is given
        Ri = options.initR
        sumgh = np.zeros((3, 1))
        for i in range(n):
            sumgh = sumgh + np.matmul((F[i] - np.identity(3)), np.matmul(Ri, P.T[i]))
        ti = np.matmul(t_factor, sumgh)
        # calculate error
        Qi = xform(P, Ri, ti)
        old_err = 0
        for i in range(n):
            vec = np.matmul(np.identity(3) - F[i], Qi.T[i])
            old_err = old_err + np.dot(vec, vec)
        new_err = old_err
    else:  # no initial guess; use weak-perspective approximation
        Ri, ti, Qi, new_err = abskernel(P, Q, F, t_factor, options["method"])
        it = 1

    # compute next pose estimate
    old_err = 2 * new_err
    while new_err > options["epsilon"] and abs((old_err - new_err) / old_err) > options["tol"]:
        old_err = new_err
        Ri, ti, Qi, new_err = abskernel(P, Qi, F, t_factor, options["method"])
        it += 1
    if new_err > options["epsilon"]:
        ghflag = 1
    # update the R/t/object-space error
    R = Ri
    t = ti
    obj_err = math.sqrt(new_err / n)

    if t[2] < 0:
        R, t = -R, -t
    t = t - np.reshape(np.matmul(Ri, pbar), (3, 1))
    if nargout < 5:
        return R, t, it, obj_err
    else:
        # calculate image-space error
        Qproj = xformproj(P, Ri, ti)
        img_err = 0
        for i in range(n):
            vec = Qproj.T[i] - Qp.T[i]
            img_err += np.dot(vec, vec)
        img_err = math.sqrt(img_err / n)
        if nargout == 5:
            return R, t, it, obj_err, img_err
        else:
            return R, t, it, obj_err, img_err, ghflag


def absolute_xy(I, T):
    """通过decode获到截图图像中心点O的绝对位置."""
    m, n = 3, 2
    px, py = 9, 8
    iter_root = pathlib.Path(__file__).parent
    img_folder = iter_root / 'image_sim'
    img_pattern = img_folder / 'test.png'
    remapped_filename = img_folder / 'remapped.png'
    rotated_filename = img_folder / 'rotated_img.png'
    A, x, y = decode.find_array(img_pattern)
    A = decode.find_A(px, py, A)
    imageio.imwrite(remapped_filename, I)
    decode.my_rotate(str(remapped_filename))
    row_grid, col_grid = decode.decode(A, rotated_filename, px, py, m, n)
    abs_x, abs_y = (row_grid - 1) * T, (col_grid - 1) * T
    return abs_x, abs_y


def calculate_center_phase(I, T):
    """计算截取图I中心点O在所在pitch内的相对位置."""
    data = I - np.average(I)
    N, M = data.shape
    center = [math.ceil((N + 1) / 2), math.ceil((M + 1) / 2)]
    phase1, phase2, u1, v1, u2, v2, row1, col1, row2, col2, theta, spectrum = cqr.SS2D(data)
    angle1 = np.angle(spectrum[row1[1] - 1][col1[1] - 1])
    angle2 = np.angle(spectrum[row2[1] - 1][col2[1] - 1])
    U1 = math.pi * (row1[1] - center[0]) * (N - 1) / N + math.pi * (col1[1] - center[1]) * (M - 1) / M
    U2 = math.pi * (row2[1] - center[0]) * (N - 1) / N + math.pi * (col2[1] - center[1]) * (M - 1) / M
    phase_1t = (angle1 + U1) % (2 * math.pi)
    phase_2t = (angle2 + U2) % (2 * math.pi)
    rel_x = phase_1t / (2 * math.pi) * T
    rel_y = phase_2t / (2 * math.pi) * T
    return rel_x, rel_y


def main():
    start_time = time.time()
    iter_root = pathlib.Path(__file__).parent
    img_folder = iter_root / 'image_sim'
    img_rgb = mpimg.imread(img_folder / '9.bmp')
    # img_gray = rgb2gray(img_rgb)
    img = cv2.normalize(img_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    # imageio.imwrite(img_folder / "test.bmp", img)

    # camera parameters
    f = 5500  # unit:um
    p_num = 200  # unit:pixel, 扫描图片的窗口大小
    p_size = 10  # unit:um, pixel size

    # pattern parameters
    T = 100  # unit:um, pitch size

    tx0, ty0, tz0, alpha0, beta0, theta0 = cqr.SS6D(img, T, f, p_size, p_num)
    beta_degree0 = alpha0 * 180 / math.pi  # unit: degree
    alpha_degree0 = beta0 * 180 / math.pi  # unit: degree
    Q, P, I = coordinate.coordinate(img, alpha_degree0, beta_degree0, T, f, p_size)
    # plt.imshow(I, cmap='gray')
    # plt.show()

    # PnP_OI
    options = {
        "tol": 1e-5,
        "epsilon": 1e-8,
        "method": 'SVD',
    }
    R, t, num_itr, obj_err, img_err, ghflag = objpose(P, Q, options, 6)
    # Finds Euler angles from a 3x3 rotation matrix
    angles = []
    eta = math.sqrt(R[0][0] ** 2 + R[1][0] ** 2)
    angles.append(math.asin(R[1][0] / eta) * 180 / math.pi)
    angles.append(math.asin(-R[2][0]) * 180 / math.pi)
    angles.append(math.asin(R[2][1] / eta) * 180 / math.pi)

    # 6 - DOF Output
    alpha_est = angles[2]  # unit:degree
    beta_est = -angles[1]  # unit:degree
    theta_est = 90 - angles[0]  # unit:degree
    tx_est = t[0][0]  # unit:um
    ty_est = t[1][0]  # unit:um
    tz_est = t[2][0]  # unit:um
    print("num of iterations: ", num_itr)
    print(alpha_est, beta_est, theta_est)
    print(tx_est, ty_est, tz_est)
    end_time = time.time()
    print("runtime: {}".format(end_time - start_time))

    # Obtain the absolute pitch position of center point O of the screenshot image I
    abs_x, abs_y = absolute_xy(I, T)
    rel_x, rel_y = calculate_center_phase(I, T)
    real_x_est = abs_x + rel_x - tx_est
    real_y_est = abs_y + rel_y - ty_est
    print("real x, y is estimated as ({}, {})".format(real_x_est, real_y_est))


if __name__ == '__main__':
    main()
