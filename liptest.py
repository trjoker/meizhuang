import cv2
import dlib
import numpy as np
from math import e, sqrt, pi
import random

# b,g,r
lipcolor = (57., 40., 207.)
white = (255., 255., 255.)
# 人像文件
FileName = 'test3.jpg'
subject = cv2.imread(FileName)
gray = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY)
# 初始化
detector = dlib.get_frontal_face_detector()
# 获取人脸检测器
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)


# dlib获取
def getPointsFromDlib(image):
    # dlib获取

    dets = detector(gray, 1)
    face_landmarks_list = []
    points = []
    for face in dets:
        shape = predictor(gray, face)  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            points.append(pt_pos)
        face_landmarks = {'chin': points[0:17], 'left_eyebrow': points[36:42],
                          'right_eyebrow': points[42:48], 'nose_bridge': points[27:31],
                          'nose_tip': points[31:36], 'left_eye': points[36:42],
                          'right_eye': points[42:48], 'top_lip': points[48:55] + points[60:65],
                          'bottom_lip': points[54:60] + [points[48]] + [points[60]] + [
                              points[60]] + [points[67]] + [points[66]] + [points[65]] + [
                                            points[64]]}
        face_landmarks_list.append(face_landmarks)
    return face_landmarks_list[0], points


def getPointsFromFile(file):
    f = open('points.txt', 'r')
    dict = f.read()
    face_landmarks = eval(dict)
    f.close()
    return face_landmarks


def showImg(name, file):
    cv2.imshow(name, file)
    cv2.imshow('original', subject)
    cv2.waitKey(0)


def nothing(x):
    pass


#获取嘴唇区域为白色的原图
def getLipmap():
    #通过
    pass

face_landmarks, points = getPointsFromDlib(FileName)
lip_map = np.zeros(subject.shape, dtype=subject.dtype)
warped_target = np.zeros(subject.shape, dtype=subject.dtype)
C2 = cv2.convexHull(np.array(points[48:62]))
cv2.drawContours(lip_map, [C2], -1, (255, 255, 255), -1)
C2 = cv2.convexHull(np.array(points[60:67]))
cv2.drawContours(lip_map, [C2], -1, (0, 0, 0), -1)




C2 = cv2.convexHull(np.array(points[48:62]))
cv2.drawContours(warped_target, [C2], -1, lipcolor, -1)
C2 = cv2.convexHull(np.array(points[60:67]))
cv2.drawContours(warped_target, [C2], -1, (0, 0, 0), -1)

l_E, _, _ = cv2.split(cv2.cvtColor(warped_target, cv2.COLOR_BGR2LAB))
l_I, _, _ = cv2.split(cv2.cvtColor(subject, cv2.COLOR_BGR2LAB))

l_E_sum = 0
l_E_sumsq = 0
l_I_sum = 0
l_I_sumsq = 0
lip_pts = []

for y in range(0, lip_map.shape[0]):
    for x in range(0, lip_map.shape[1]):
        # print(lip_map[y][x])
        if (lip_map[y][x][2] != 0):
            l_E_sum += l_E[y, x]  # calculating mean for only lip area
            l_E_sumsq += l_E[y, x] ** 2  # calculating var for only lip area
            l_I_sum += l_I[y, x]  # calculating mean for only lip area
            l_I_sumsq += l_I[y, x] ** 2  # calculating var for only lip area
            lip_pts.append([y, x])
l_E_mean = l_E_sum / len(lip_pts)
l_I_mean = l_I_sum / len(lip_pts)

# 这里因为是写死的颜色，标准差为0
l_E_std = sqrt((l_E_sumsq / len(lip_pts)) - l_E_mean ** 2)
l_I_std = sqrt((l_I_sumsq / len(lip_pts)) - l_I_mean ** 2)
if l_E_std == 0:
    l_E_std = 1
l_E = (l_I_std / l_E_std * (
    l_E - l_E_mean)) + l_I_mean


def Gauss(x):
    return e ** (-0.5 * float(x))


M = cv2.cvtColor(subject, cv2.COLOR_BGR2LAB)
warped_target_LAB = cv2.cvtColor(warped_target, cv2.COLOR_BGR2LAB)
counter = 0

sample = lip_pts.copy()
random.shuffle(sample)
avg_maxval = 0
for p in lip_pts:
    q_tilda = 0
    maxval = -1
    counter += 1
    print(counter / len(lip_pts) * 100, " %")
    # for i in range(0, 500):
    for i in range(0, len(sample)):
        q = sample[i]
        curr = (Gauss(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) / 5) * Gauss(
            ((float(l_E[q[0]][q[1]]) - float(l_I[p[0]][p[1]])) / 255) ** 2))
        if maxval < curr:
            maxval = curr
            q_tilda = q
            if maxval >= 0.9:
                break

    avg_maxval += maxval
    print("max = ", maxval)
    M[p[0]][p[1]] = warped_target_LAB[q_tilda[0]][q_tilda[1]]
# cv2.imshow('M', cv2.cvtColor(M, cv2.COLOR_LAB2BGR))
print("avgmax = ", avg_maxval / len(lip_pts))

output = cv2.cvtColor(subject.copy(), cv2.COLOR_BGR2LAB)
for p in lip_pts:
    output[p[0]][p[1]][1] = M[p[0]][p[1]][1]
    output[p[0]][p[1]][2] = M[p[0]][p[1]][2]

output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)
cv2.imshow('out', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
