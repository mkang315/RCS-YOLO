import os
import sys
import glob
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from data.data_config import get_train_data
from PIL import Image

import json
# 转换成四点坐标
import math

def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(int(xc + pResx)), str(int(yc + pResy))

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x * dw, 2)
    w = round(w * dw, 2)
    y = round(y * dh, 2)
    h = round(h * dh, 2)
    return (x, y, w, h)

def progress(percent, width=100):
    if percent > 1:
        percent = 1
    show_str = ('[%%-%ds]' % width) % (int(percent * width) * '#')
    print('\r%s %s%%' % (show_str, int(percent * 100)), end='', file=sys.stdout, flush=True)

def get_ellipse_param(major_radius, minor_radius, angle):
    a, b = major_radius, minor_radius
    sin_theta = np.sin(-angle)
    cos_theta = np.cos(-angle)
    A = a ** 2 * sin_theta ** 2 + b ** 2 * cos_theta ** 2
    B = 2 * (a ** 2 - b ** 2) * sin_theta * cos_theta
    C = a ** 2 * cos_theta ** 2 + b ** 2 * sin_theta ** 2
    F = -a ** 2 * b ** 2
    return A, B, C, F

def calculate_rectangle(A, B, C, F):
    '''
    椭圆上下外接点的纵坐标值
    '''
    y = np.sqrt(4 * A * F / (B ** 2 - 4 * A * C))
    y1, y2 = -np.abs(y), np.abs(y)

    '''
    椭圆左右外接点的横坐标值
    '''
    x = np.sqrt(4 * C * F / (B ** 2 - 4 * C * A))
    x1, x2 = -np.abs(x), np.abs(x)

    return (x1, y1), (x2, y2)

def get_rectangle(major_radius, minor_radius, angle, center_x, center_y):
    A, B, C, F = get_ellipse_param(major_radius, minor_radius, angle)
    p1, p2 = calculate_rectangle(A, B, C, F)
    return center_x + p1[0], center_y + p1[1], center_x + p2[0], center_y + p2[1]

def convert_annotation(path, fold):
    filnamess = "annotations_train.json"
    if fold == "valdata":
        filnamess = "annotations_val.json"
    with open(os.path.join(path, fold, filnamess), 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        for item in json_data.items():
            filename = item[1]['filename']
            for regin in item[1]['regions']:
                tm = regin['shape_attributes']
                if tm['name'] == 'ellipse':
                    cx = float(tm['cx'])
                    cy = float(tm['cy'])
                    w = float(tm['rx'])
                    h = float(tm['ry'])
                    angle = float(tm['theta'])
                    x0, y0, x2, y2 = get_rectangle(w, h, angle, cx, cy)
                elif tm['name'] == 'polygon':
                    xx = tm['all_points_x']
                    yy = tm['all_points_y']
                    x0, y0 = min(xx), min(yy)
                    x2, y2 = max(xx), max(yy)
            out_file = open(get_train_data() + fold + '/%s' % (filename.replace(".jpg", ".txt")), 'w')
            img = Image.open(os.path.join(get_train_data(), fold, filename))
            w, h = img.width, img.height
            b = (int(x0), int(x2), int(y0), int(y2))
            print(b)
            bb = convert((w, h), b)
            out_file.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == "__main__":
    current = 0
    for fold in ["traindata", "valdata"]:
        current += 1
        convert_annotation(get_train_data(), fold)
