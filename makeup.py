import cv2
import face_recognition
import dlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
# 人脸分类器
from scipy.stats._discrete_distns import poisson_gen

# 初始化
detector = dlib.get_frontal_face_detector()
# 获取人脸检测器
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)
fileName = 'test.png'
img = cv2.imread(fileName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


# face recongnition获取
def getPointsFromFaceRecongnition(image):
    image = face_recognition.load_image_file("test.png")
    face_landmarks_list = face_recognition.face_landmarks(image)
    return face_landmarks_list


def drawPoints(name=None):
    face_landmarks_list, points = getPointsFromDlib('test.png')
    # face_landmarks_list = getPointsFromFaceRecongnition('test.png')
    for face_landmarks in face_landmarks_list:
        if (name == None):
            for point in points:
                cv2.circle(img, point, 2, (0, 255, 0), 1)
        else:
            for point in face_landmarks[name]:
                cv2.circle(img, point, 2, (0, 255, 0), 1)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 获取某一区域像素点
def getPixels():
    image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    face_landmarks_list, points = getPointsFromDlib(fileName)
    # cv2.drawContours(img, face_landmarks_list[0]['top_lip'], 10, (0, 0, 255), 3)
    cv2.drawContours(img, face_landmarks_list[0]['top_lip'], 10, (0, 0, 255), 3)
    plt.figure()
    plt.imshow(img)
    plt.show()


# 获取边界
def getContours():
    # 创建和原图同样大小的黑图
    image = np.zeros((img.shape), dtype=np.uint8)
    face_landmarks_list, points = getPointsFromDlib(fileName)
    a = np.array([face_landmarks_list[0]['top_lip']])
    cv2.fillPoly(image, a, (255, 255, 255))
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    image2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, 0, (0, 0, 255), 3)
    cv2.imshow("img", image)
    cv2.waitKey(0)


# 绘制多边形
def drawpolygon():
    image = np.zeros((img.shape), dtype=np.uint8)
    face_landmarks_list, points = getPointsFromDlib(fileName)
    top_lip = np.array([face_landmarks_list[0]['top_lip']])
    cv2.fillPoly(image, top_lip, (0, 0, 230))
    bitwiseAnd = cv2.bitwise_and(img, image)
    cv2.imshow("img", bitwiseAnd)
    cv2.waitKey(0)
