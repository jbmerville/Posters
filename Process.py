import random
import cv2
import matplotlib.pyplot as plt
from matplotlib import figure


class Process:
    @staticmethod
    def PrepareIMG(img):
        # permet de convertir l'image en nuance de gris
        greyIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return greyIMG

    def GetContours(img):
        ret, thresh = cv2.threshold(img, 100, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(str(len(contours))+" shapes found")
        return contours

    @staticmethod
    def GetRandomColor():
        R = random.randrange(0, 255)
        G = random.randrange(0, 255)
        B = random.randrange(0, 255)
        return (R, G, B)

    def DrawOneContours(inputIMG, contours, index: int, colorchoice):

        # mask=np.zeros(inputIMG.shape,np.uint8)
        img = cv2.drawContours(inputIMG, contours, index,
                               colorchoice, thickness=cv2.FILLED)
        #img = cv2.drawContours(mask, contours,index, (249,255,10),thickness=cv2.FILLED)

        return img

    def PltIMG(img, title="", size=None, cm=None):
        plt.figure(figsize=(size, size))
        plt.imshow(img, cmap=cm)
        plt.title(title)
        plt.show()
