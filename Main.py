import IMG
from IMG import IMG as I
from Process import Process as P
from ColorsIMG import Color as C
import numpy as np


# Main class


def Main():
    imgName = "BlackML.jpg"
    imgSize = 500
    inputIMG = I.getIMG(imgName)
    greyIMG = P.PrepareIMG(inputIMG)
    #P.PltIMG(greyIMG, "GreyScaleIMG", 7, "gray")

    contours = P.GetContours(greyIMG)
    CIMG = inputIMG
    Colors = C.ToulouseStyle()

    CIMG = I.getBackgroundimage(inputIMG, Colors.BackgroundColor)

    for i in range(0, len(contours)):

        color = C.PickColor(Colors.ColorList())
        CIMG = P.DrawOneContours(CIMG, contours, i, color)

    P.PltIMG(CIMG, "Contour", 7)


if __name__ == "__main__":

    Main()
