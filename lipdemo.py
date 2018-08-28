import cv2
import face_recognition
import dlib
import matplotlib.pyplot as plt
import numpy as np
import makeup

# b,g,r
lipcolor = (57., 40., 207.)
white = (255., 255., 255.)
# 人像文件
FileName = 'test6.jpg'
oriImg = cv2.imread(FileName)
gray = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
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
    return face_landmarks_list, points


def showImg(name, file):
    cv2.imshow(name, file)
    cv2.imshow('oriimg', oriImg)
    cv2.waitKey(0)


def nothing(x):
    pass


# 获取嘴唇区域坐标点
face_landmarks_list, points = getPointsFromDlib(FileName)
top_lip = np.array([face_landmarks_list[0]['top_lip']])
bottom_lip = np.array([face_landmarks_list[0]['bottom_lip']])

# 将原图嘴唇区域扣掉
imageMask = np.zeros((oriImg.shape), dtype=np.uint8)
cv2.fillPoly(imageMask, top_lip, lipcolor)
cv2.fillPoly(imageMask, bottom_lip, lipcolor)

gray = cv2.cvtColor(imageMask, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# 嘴唇部分是黑色的背景
img_bg = cv2.bitwise_and(oriImg, oriImg, mask=mask_inv)

# 获取只有嘴唇的黑底图片
imageBlack = np.zeros((oriImg.shape), dtype=np.uint8)
cv2.fillPoly(imageBlack, top_lip, white)
cv2.fillPoly(imageBlack, bottom_lip, white)
lipimg = cv2.bitwise_and(imageBlack, oriImg)


# 使用addWeighted方式
def addWeighted():
    # 混合嘴唇
    dst = cv2.addWeighted(imageMask, 0.2, lipimg, 0.2, 0)
    # 与原图合成
    result = cv2.add(dst, img_bg)
    cv2.namedWindow('result')
    cv2.createTrackbar('alpha', 'result', 0, 10, nothing)
    cv2.createTrackbar('beta', 'result', 0, 10, nothing)
    cv2.createTrackbar('gamma', 'result', 0, 20, nothing)
    switch = '0:OFF\n1:ON'
    cv2.createTrackbar(switch, 'result', 0, 1, nothing)
    while (1):

        cv2.imshow('result', result)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        alpha = cv2.getTrackbarPos('alpha', 'result')
        beta = cv2.getTrackbarPos('beta', 'result')
        gamma = cv2.getTrackbarPos('gamma', 'result')
        s = cv2.getTrackbarPos(switch, 'result')
        if s == 0:
            pass
        else:
            dst = cv2.addWeighted(imageMask, alpha / 10, lipimg, beta / 10, gamma)
            result = cv2.add(dst, img_bg)
    cv2.destroyAllWindows()


# 将原嘴唇的 bgr -》yuv
# 嘴唇颜色color -》 yuv
# 使用原嘴唇的y 加上 唇彩的uv
def yuv():
    lipyuv = cv2.cvtColor(lipimg, cv2.COLOR_BGR2YUV)
    maskyuv = cv2.cvtColor(imageMask, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(lipyuv)
    y2, u2, v2 = cv2.split(maskyuv)
    img = cv2.merge([y, u2, v2])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    result = cv2.add(img,img_bg)
    showImg('result',result)

# 将原嘴唇的 bgr -》hsv
# 嘴唇颜色color -》 hsv
def hsv():
    lip = cv2.cvtColor(lipimg, cv2.COLOR_BGR2HSV)
    mask = cv2.cvtColor(imageMask, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(lip)
    h2, s2, v2 = cv2.split(mask)
    img = cv2.merge([h2, s2, v])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    result = cv2.add(img,img_bg)
    showImg('result',result)



hsv()
