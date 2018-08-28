import cv2
import dlib

path = "test.png"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 人脸分类器
detector = dlib.get_frontal_face_detector()
# 获取人脸检测器
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

dets = detector(gray, 1)
for face in dets:
    shape = predictor(img, face)  # 寻找人脸的68个标定点
    # 遍历所有点，打印出其坐标，并圈出来
    points = []
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        points.append(pt_pos)
        # cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
        # print(pt_pos)
    print(points[48:69])
    print(len(points[48:69]))
    for point in points[
                 48:68]:  # [0:17]脸颊   [17:27]眉毛 [27 :36]鼻子 [36:48]眼睛  [48:68] 嘴唇  [48:60] 嘴唇外圈  [60:68]嘴唇内圈
        cv2.circle(img, point, 2, (0, 255, 0), 1)
    print(len(points))
    cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
