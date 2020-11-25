
import os
import cv2
import numpy as np


class IMG:
    @staticmethod
    def getIMG(path: str):
        path = os.path.expanduser("~/Desktop/Art/"+path)
        img = cv2.imread(path)
        print(img)
        return img

    @ staticmethod
    def showIMG(img, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = img.shape[:2]
        if width is None and height is None:
            imR = img
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        imR = cv2.resize(img, dim, interpolation=inter)

        cv2.imshow('', imR)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    @staticmethod
    def getBackgroundimage(inputIMG, Color=(0, 0, 0)):
        h, w, d = inputIMG.shape
        image = np.zeros((h, w, 3), np.uint8)
        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(Color))
        # Fill image with color
        image[:] = color
        return image
