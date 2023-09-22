import numpy as np
import cv2
from PIL import Image
import math
import pathlib
from scipy import ndimage
import imageio


def find_array(image_name):
    """
    This function can find the array represented by the image

    Input: image_name: file name of the image captured by the camera
    Output: my_array: An array represented by "image_name"
            pos_0_row, pos_0_col: The position of the image center
    """
    img = Image.open(image_name)
    a = np.array(img.convert('P', palette=Image.ADAPTIVE, colors=2))

    h, w = a.shape
    t1 = np.ones(w)
    t2 = np.ones(h)

    heng = []
    for i in range(h):
        if (t1 == a[i]).all():
            heng.append(i)

    shu = []
    for i in range(w):
        if (t2 == a[:, i]).all():
            shu.append(i)

    num_heng = len(heng)
    num_shu = len(shu)
    num_row = 0

    pos_0_row = 0
    pos_0_col = 0

    for i in range(num_heng - 1):
        if heng[i + 1] > int(h / 2) >= heng[i]:
            pos_0_row = num_row
        if heng[i + 1] != heng[i] + 1:
            num_row = num_row + 1

    num_col = 0
    for i in range(num_shu - 1):
        if shu[i + 1] > int(w / 2) >= shu[i]:
            pos_0_col = num_col
        if shu[i + 1] != shu[i] + 1:
            num_col = num_col + 1

    a = a[heng[0]:heng[-1], shu[0]:shu[-1]]

    my_array = cv2.resize(a, (num_col, num_row))
    return my_array, pos_0_row, pos_0_col


def my_decode(A, M):
    """
    This function can decode M from A
    Input: A, M
    Output: rst_i, rst_j: the position of the left_upper corner of the subarray M
    """
    r, s = A.shape
    m, n = M.shape
    rst_i = 0
    rst_j = 0
    for i in range(r - m + 1):
        for j in range(s - n + 1):
            if (M == A[i:i + m, j:j + n]).all():
                rst_i = i
                rst_j = j
                break
    return rst_i, rst_j


def find_subarray(px, py, m, n, test_array):
    """
    This function can find one subarray of shape (m,n) from test_array

    Input: px, py, m, n, test_array
    Output: rst_array: subarray of shape (m, n)
            x, y: position of the left_upper corner of the subarray
    """
    my_array = test_array.copy()
    pos = np.where(my_array == 1)
    h, w = my_array.shape
    pos_row = pos[0][0]
    pos_col = pos[1][0]

    x = pos_row - px * int(pos_row / px)
    y = pos_col - py * int(pos_col / py)

    delete_row = []
    for i in range(h):
        if abs(i - pos_row) % px != 0:
            delete_row.append(i)
    my_array = np.delete(my_array, delete_row, axis=0)

    delete_col = []
    for i in range(w):
        if abs(i - pos_col) % py != 0:
            delete_col.append(i)
    my_array = np.delete(my_array, delete_col, axis=1)
    rst_array = my_array[:m, :n]
    return rst_array, x, y


def find_A(px, py, test_array):
    """
    This function can find the whole map A from test_array

    Input: px, py, m, n, test_array
    Output: my_array
    """
    my_array = test_array.copy()
    pos = np.where(my_array == 1)
    h, w = my_array.shape
    pos_row = pos[0][0]
    pos_col = pos[1][0]

    delete_row = []
    for i in range(h):
        if abs(i - pos_row) % px != 0:
            delete_row.append(i)
    my_array = np.delete(my_array, delete_row, axis=0)

    delete_col = []
    for i in range(w):
        if abs(i - pos_col) % py != 0:
            delete_col.append(i)
    my_array = np.delete(my_array, delete_col, axis=1)
    return my_array


def decode(A, image_name, px, py, m, n):
    """
    Input: A: The whole grid map
           image_name: Image file captured by the camera
           px, py: Number of pixels
           m, n: Shape of the subarray
    Output: i, j: The position of the center of the image (O)
    """
    my_array, pos_0_row, pos_0_col = find_array(image_name)
    M, x, y = find_subarray(px, py, m, n, my_array)
    rst_i, rst_j = my_decode(A, M)
    pos_i = px * (rst_i + 1) + pos_0_row - x
    pos_j = py * (rst_j + 1) + pos_0_col - y
    return pos_i, pos_j


def my_rotate(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    x1, x2, y1, y2 = 0, 0, 0, 0

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

    if x1 == x2:
        rotate_angle = 0
    else:
        t = float(y2 - y1) / (x2 - x1)
        rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle)

    h, w = np.shape(rotate_img)[0], np.shape(rotate_img)[1]
    rotate_img = rotate_img[int(h / 20):-int(h / 20), int(w / 20):-int(w / 20), :]
    iter_root = pathlib.Path(__file__).parent
    img_folder = iter_root / 'image_sim'
    imageio.imwrite(img_folder / "rotated_img.png", rotate_img)
