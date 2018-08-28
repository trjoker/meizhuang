import numpy as np
import cv2
from matplotlib import pyplot as plt


# opencv 读取图像 matplotlib显示
def rgbtest():
    img = cv2.imread('test.png')
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()
    cv2.imshow('bgr image', img)
    cv2.imshow('rgb image', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drawLine():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.line(img, (0, 0), (512, 512), (0, 0, 255), 10)
    cv2.imshow('line', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drawRect():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.rectangle(img, (0, 0), (512, 512), (0, 0, 255), 10)
    cv2.imshow('rect', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drawCircle():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.circle(img, (300, 300), 63, (0, 0, 255), -1)
    cv2.imshow('circle', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drawEllipse():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
    cv2.imshow('circle', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drawLines():
    img = np.zeros((512, 512, 3), np.uint8)
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 255))
    cv2.imshow('lines', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = np.zeros((512, 512, 3), np.uint8)

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1


def mouseClickCricle():
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while (1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xff == 27:
            break
    cv2.destroyAllWindows()


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    color = (b, g, r)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                # pass
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
                cv2.circle(img, (x, y), 5, color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # if mode == True:
        #     cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
        # else:
        #     cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


def mouseDraw():
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):  # 切换模式
            global mode
            mode = not mode
        elif k == 27:
            break
    cv2.destroyAllWindows()


def nothing(x):
    pass


def changePixel():
    img = cv2.imread('test.png')
    img[101, 100] = [0, 0, 255]
    img[100, 101] = [0, 0, 255]
    img[101, 101] = [0, 0, 255]
    print(img.item(10, 10, 2))
    img.itemset((10, 10, 2), 100)
    print(img.item(10, 10, 2))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def roi():
    img = cv2.imread('test.png')
    print(img.shape)
    ball = img[300:320, 150:170]
    img[50:70, 50:70] = ball
    img = cv2.imshow('test', img)
    cv2.waitKey(0)


def border():
    BLUE = [255, 0, 0]
    img1 = cv2.imread('test.png')
    replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)
    plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
    plt.show()


def add():
    img1 = cv2.imread('test2.jpg')
    img2 = cv2.imread('test3.jpg')
    h, w, _ = img1.shape
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    dst = cv2.addWeighted(img1, 0.2, img2, 0.8, 0)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show(img_name, img_data):
    cv2.imshow(img_name, img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def imgand():
    img1 = cv2.imread('test2.jpg')
    img2 = cv2.imread('logo.jpg')

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]
    cv2.imshow('roi',roi)
    cv2.imshow('logo',img2)
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',img2gray)
    #img2gray = cv2.bitwise_not(img2gray)
    #ret, mask = cv2.threshold(img2gray, 50, 255, cv2.THRESH_BINARY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    cv2.imshow('mask',mask)
    cv2.imshow('mask_inv',mask_inv)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    cv2.imshow('bg',img1_bg)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
    cv2.imshow('fg',img2_fg)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)

    img1[0:rows, 0:cols ] = dst
    cv2.imshow('dst',dst)
    cv2.imshow('result',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imgand()