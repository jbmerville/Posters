
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Colors import InputColorType
from PIL import Image as PILImg


class Image:
    def __init__(self, path: str = None, img=None) -> None:
        if not path and img is None:
            raise Exception("❌ Constructor and img are both None." % path)
        elif not path:
            self.img = img
        elif not os.path.exists(path):
            raise Exception("❌ Path: \"%s\" does not exit." % path)
        else:
            self.img = cv2.imread(path)

    def display(self, width: int = None, height: int = None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = self.img.shape[:2]
        imageResized = self.img
        if width is None:  # explique le calcule
            r = height / float(h)
            dim = (int(w * r), height)
        else:  # de meme
            r = width / float(w)
            dim = (width, int(h * r))

        imageResized = cv2.resize(self.img, dim, interpolation=inter)
        cv2.imshow('', imageResized)
        cv2.waitKey(0)  # pourquoi 0?
        cv2.destroyAllWindows()

    def createMonoColorImage(self, Color=(255, 255, 255)):
        h, w, d = self.img.shape
        image = np.zeros((h, w, 3), np.uint8)  # create np array to represent the image
        color = tuple(Color)  # RGB tuple --> CV2 color tuple
        image[:] = color  # fill image with color
        return Image(img=image)

    def createMonoColorImageFactor(self, factor, Color=(255, 255, 255)):
        h, w, d = self.img.shape
        image = np.zeros((h*factor, w*factor, 3), np.uint8)  # create np array to represent the image
        color = tuple(Color)  # RGB tuple --> CV2 color tuple
        image[:] = color  # fill image with color
        return Image(img=image)

    def getGreyScale(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def getMaskfromColor(self, minColor, maxColor):
        HSV = self.getHSV()
        return cv2.inRange(HSV, minColor, maxColor)

    def getImagefromColor(self, ColorType: InputColorType):
        if ColorType == InputColorType.FOREST:
            # light green
            minColor = (40, 40, 40)
            maxColor = (70, 255, 255)
            mask = self.getMaskfromColor(minColor, maxColor)
            NeutralIMG = self.createMonoColorImage()
            return NeutralIMG.createMaskImageInverted(mask)
        elif ColorType == InputColorType.WATER:
            # light blue
            minColor = (80, 50, 50)
            maxColor = (100, 255, 255)
            mask = self.getMaskfromColor(minColor, maxColor)
            NeutralIMG = self.createMonoColorImage()
            return NeutralIMG.createMaskImageInverted(mask)
        elif ColorType == InputColorType.HOUSE:
           # red
            minColor = (0, 50, 50)
            maxColor = (10, 255, 255)
            mask = self.getMaskfromColor(minColor, maxColor)
            NeutralIMG = self.createMonoColorImage()
            return NeutralIMG.createMaskImageInverted(mask)
        elif ColorType == InputColorType.AIRPORT:
            # orange
            minColor = (15, 200, 200)
            maxColor = (18, 255, 255)
            mask = self.getMaskfromColor(minColor, maxColor)
            NeutralIMG = self.createMonoColorImage()
            return NeutralIMG.createMaskImageInverted(mask)
        elif ColorType == InputColorType.RAIL:
            # brown
            minColor = (16, 220, 100)
            maxColor = (18, 255, 150)
            mask = self.getMaskfromColor(minColor, maxColor)
            NeutralIMG = self.createMonoColorImage()
            return NeutralIMG.createMaskImageInverted(mask)

        elif ColorType == InputColorType.MANMADE:
            # white
            self.img = np.invert(self.img)
            minColor = (0, 0, 0)
            maxColor = (0, 0, 0)

            mask = self.getMaskfromColor(minColor, maxColor)
            NeutralIMG = self.createMonoColorImage()
            return NeutralIMG.createMaskImageInverted(mask)
        elif ColorType == InputColorType.ROAD:

            self.img = np.invert(self.img)
            minColor = (0, 0, 0)
            maxColor = (0, 0, 0)
            # maxColor = (100, 250, 250)
            mask = self.getMaskfromColor(minColor, maxColor)
            NeutralIMG = self.createMonoColorImage((255, 255, 255))
            return NeutralIMG.createMaskImageInverted(mask)

    def createMaskImageInverted(self, maskCOLOR):
        img = cv2.bitwise_or(self.img, self.img, mask=maskCOLOR)
        return np.invert(img)

    def minlimitColor(self, color):

        color = np.uint8([[[color[0], color[1], color[2]]]])
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        return (int(max(hsv[0][0][0]-10, 0)), int(100), int(100))

    def maxlimitColor(self, color):
        color = np.uint8([[[color[0], color[1], color[2]]]])
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        return (int(min(hsv[0][0][0]+10, 255)), int(255), int(55))

    def createMaskImage(self, maskCOLOR):
        return cv2.bitwise_or(self.img, self.img, mask=maskCOLOR)

    def converttoBlackWhite(self):
        img_gray = self.getGreyScale()
        thresh, im_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        imgBW = Image("", im_bw)
        self.img = imgBW.img

    def getHSV(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    def revertImage(self):
        self.img = np.invert(self.img)
        return self.img

    def getShapes(self, allPoints=False):
        greyScale = self.getGreyScale()
        # ret, thresh = cv2.adaptiveThreshold(greyScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)[-2:]

        ret, thresh = cv2.threshold(greyScale, 100, 255, cv2.THRESH_BINARY_INV)
        if allPoints:
            return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        else:
            return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    def colorShape(self, shape, color):
        self.img = cv2.drawContours(self.img, [shape], 0, color, thickness=cv2.FILLED)

    def colorShapeAreaMin(self, shape, color, threshold):
        area = cv2.contourArea(shape)
        if area < threshold:
            self.img = cv2.drawContours(self.img, [shape], 0, color, thickness=cv2.FILLED)

    def getArea(self, shape):
        return cv2.contourArea(shape)

    def colorShapeAreaMax(self, shape, color, threshold):
        area = cv2.contourArea(shape)
        if area > threshold:
            self.img = cv2.drawContours(self.img, [shape], 0, color, thickness=cv2.FILLED)

    def colorContour(self, shape, color):
        self.img = cv2.drawContours(self.img, [shape], 0, color, thickness=5)

    def colorContourAreaMax(self, shape, color, threshold):
        area = cv2.contourArea(shape)
        if area > threshold:
            self.img = cv2.drawContours(self.img, [shape], 0, color, thickness=5)

    def hullContour(self, shape):
        return cv2.convexHull(shape)

    def colorShapes(self, shapes, color):
        for i in range(len(shapes)):
            self.img = cv2.drawContours(self.img, shapes, i, color, thickness=cv2.FILLED)

    def scaleUp(self):
        self.img = cv2.pyrUp(self.img)

    def scaleDown(self):
        self.img = cv2.pyrDown(self.img)

    def removeNoise(self, level):

        self.revertImage()
        self.scaleUp()
        kernel = np.ones((level, level), np.uint8)
        # self.img = cv2.morphologyEx(self.img, cv2.MORPH_CROSS, kernel)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        self.revertImage()
        self.scaleDown()

    def smooth2(self, size, iter):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        (thresh, binRed) = cv2.threshold(self.img, 128, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel, iterations=iter)
        self.img = opening

    def smooth3(self, size, iter):
        blur = ((size, size), iter)
        erode_ = (size+2, size+2)
        dilate_ = (size, size)
        self.img = cv2.dilate(cv2.erode(cv2.GaussianBlur(
            self.img, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))

    def smoothImage(self, iter, size):
        ret, thresh = cv2.threshold(self.img, 100, 255, cv2.THRESH_BINARY)
        t = Image("", thresh)

        self.img = cv2.pyrUp(t.img)

        for i in range(0, iter):
            self.img = cv2.medianBlur(self.img, size)

        self.img = cv2.pyrDown(self.img)

        ret, thresh = cv2.threshold(self.img, 200, 255, cv2.THRESH_BINARY)
        smoothedImage = Image("", thresh)
        self.img = smoothedImage.img

    def fillBlank(self, level):
        self.revertImage()
        self.scaleUp()
        kernel = np.ones((level, level), np.uint8)
        # self.img = cv2.morphologyEx(self.img, cv2.MORPH_CROSS, kernel)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        # self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        self.revertImage()
        self.scaleDown()

    def dilateStyle(self, size, iter):
        self.img = self.revertImage()
        kernel = kernel = np.ones((size, size), np.uint8)
        self.img = cv2.dilate(self.img, kernel, iterations=iter)
        self.img = self.revertImage()

    def erodeStyle(self, size, iter):
        self.img = self.revertImage()
        kernel = np.ones((size, size), np.uint8)
        self.img = cv2.erode(self.img, kernel, iterations=iter)
        self.img = self.revertImage()

    def displayV2(self, title="", size=None, cm=None):
        plt.figure(figsize=(size, size))
        plt.imshow(self.img, cmap=cm)
        plt.title(title)
        plt.show()

    def saveImage(self, Path: str):

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(Path, self.img)

    def save2PDF(self, Path: str):

        pil_image = PILImg.fromarray(self.img)

        pil_image.save(Path, quality=100)


# pour les contours

def getJoinShape(shapes, threshold):
    listsmallshapes = list()
    listothershapes = list()
    for shape in shapes:
        if cv2.contourArea(shape) < threshold:
            listsmallshapes.append(shape)
        else:
            listothershapes.append(shape)
