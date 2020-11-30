
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    def createMonoColorImage(self, Color=(0, 0, 0)):
        h, w, d = self.img.shape
        image = np.zeros((h, w, 3), np.uint8)  # create np array to represent the image
        color = tuple(reversed(Color))  # RGB tuple --> CV2 color tuple
        image[:] = color  # fill image with color
        return Image(img=image)

    def getGreyScale(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def getShapes(self):
        greyScale = self.getGreyScale()
        #ret, thresh = cv2.adaptiveThreshold(greyScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)[-2:]
        ret, thresh = cv2.threshold(greyScale, 100, 255, 0)

        return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    def colorShape(self, shape, color):
        self.img = cv2.drawContours(self.img, [shape], 0, color, thickness=cv2.FILLED)

    def colorShapes(self, shapes, color):
        for i in range(len(shapes)):
            self.img = cv2.drawContours(self.img, shapes, i, color, thickness=cv2.FILLED)

    def displayV2(self, title="", size=None, cm=None):
        plt.figure(figsize=(size, size))
        plt.imshow(self.img, cmap=cm)
        plt.title(title)
        plt.show()
