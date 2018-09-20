import cv2
import numpy as np

drawing = False  # 是否开始画图
mode = 1  # 1、上嘴唇上侧 2、上嘴唇下侧 3、下嘴唇上侧 4、下嘴唇下侧
start = (-1, -1)
top_lip_top = []
top_lip_bottom = []
bottom_lip_top = []
bottom_lip_bottom = []


def mouse_event(event, x, y, flags, param):
    global start, drawing, mode
    if event == cv2.EVENT_LBUTTONUP:
        if mode == 1:
            top_lip_top.append([x, y])
        elif mode == 2:
            top_lip_bottom.append([x, y])
        elif mode == 3:
            bottom_lip_top.append([x, y])
        elif mode == 4:
            bottom_lip_bottom.append([x, y])
        print('mode', mode, x, y)
        cv2.circle(img, (x, y), 3, (0, 0, 0), 1)


FileName = 'lip.jpg'
img = cv2.imread(FileName)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_event)
while (True):
    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        if len(bottom_lip_top) == 0:
            bottom_lip_top = top_lip_bottom
        face_landmarks = {'top_lip': top_lip_top + top_lip_bottom,
                          'bottom_lip': bottom_lip_top + bottom_lip_bottom}
        f = open('points.txt', 'w')
        f.write(str(face_landmarks))
        f.close()
        break
    elif cv2.waitKey(1) == ord('q'):
        mode = 1
        print('mode1')
    elif cv2.waitKey(1) == ord('w'):
        mode = 2
        print('mode2')
    elif cv2.waitKey(1) == ord('e'):
        mode = 3
        print('mode3')
    elif cv2.waitKey(1) == ord('r'):
        mode = 4
        print('mode4')
