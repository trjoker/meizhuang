import cv2
import numpy as np

drawing = False  # 是否开始画图
mode = 1  # 1、上嘴唇上侧 2、上嘴唇下侧 3、下嘴唇上侧 4、下嘴唇下侧
start = (-1, -1)
face_landmarks = {}


def mouse_event(event, x, y, flags, param):
    global start, drawing, mode
    if event == cv2.EVENT_LBUTTONUP:
        if (mode == 1):
            # points.append([x, y])
            pass
        cv2.circle(img, (x, y), 3, (0, 0, 0), 1)


FileName = 'lip.jpg'
img = cv2.imread(FileName)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_event)
while (True):
    cv2.imshow('image', img)
    # 按下m切换模式
    if cv2.waitKey(1) == 27:
        # print(points)
        break
    elif cv2.waitKey(1) == ord('1'):
        mode = 1
    elif cv2.waitKey(1) == ord('2'):
        mode = 2
    elif cv2.waitKey(1) == ord('3'):
        mode = 3
    elif cv2.waitKey(1) == ord('4'):
        mode = 4
