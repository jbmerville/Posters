from Image import Image
from Colors import Toulouse
from utils import getRandomColor, FakeRand
from Colors import InputColorType
from Colors import Color
from Shape import Shape
from PIL import Image as PILImg
import numpy as np
import cv2
import img2pdf


def Main(key, isPreview: bool):
    FakeRandobj = FakeRand(key)
    SavingPath = "/Users/matthieumerville/Desktop/Art/Test2/"
    name = "Lyon_Z13.5_R2"
    endName = "_Original"
    colorImage = Image("stock_pictures/%s.png" % name)

    toulouse = Toulouse()

    # split Main image into sub images
    railImage = Image("", colorImage.getImagefromColor(InputColorType.RAIL))
    natureImage = Image("", colorImage.getImagefromColor(InputColorType.FOREST))
    waterImage = Image("", colorImage.getImagefromColor(InputColorType.WATER))
    airportImage = Image("", colorImage.getImagefromColor(InputColorType.AIRPORT))
    manMadeImage = Image("", colorImage.getImagefromColor(InputColorType.MANMADE))

    # shape extraction

    # rail
    railShapes = list()
    # railImage.revertImage()
    for shape in railImage.getShapes():
        S = Shape(shape, InputColorType.RAIL)
        railShapes.append(S)
    # nature
    natureShapes = list()
    for shape in natureImage.getShapes():
        S = Shape(shape, InputColorType.FOREST)
        natureShapes.append(S)
    # water
    waterShapes = list()
    for shape in waterImage.getShapes():
        S = Shape(shape, InputColorType.WATER)
        waterShapes.append(S)
    # airport
    aeroportShapes = list()
    for shape in airportImage.getShapes():
        S = Shape(shape, InputColorType.AIRPORT)
        aeroportShapes.append(S)
    # manMade
    manMadeShapes = list()
    for shape in manMadeImage.getShapes():

        S = Shape(shape, InputColorType.MANMADE)
        manMadeShapes.append(S)

    if isPreview:  # lowQuality Image
        SavingPath += "Preview_%s_%s" % (name, endName)
        newImage = railImage.createMonoColorImage(toulouse.getRGBBackgroundColor())

        # color
        colorRail = toulouse.getRGBRailTrackColor()
        for S in railShapes:
            newImage.colorShape(S.originalshape, colorRail)

        for S in natureShapes:
            colorNature = toulouse.pickNatureColor(FakeRandobj)
            newImage.colorShape(S.originalshape, colorNature)
        colorWater = toulouse.getRGBWaterColor()

        colorAirport = toulouse.getRGBAirportColor()
        for S in aeroportShapes:
            newImage.colorShape(S.originalshape, colorAirport)
        for S in manMadeShapes:
            colorManMade = toulouse.pickManMadeColor(FakeRandobj)
            newImage.colorShape(S.getTransformedShape(), colorManMade)

        for S in waterShapes:
            newImage.colorShape(S.originalshape, colorWater)

    else:  # High Quality Image
        factorResize = 3
        factorSmooth = 5  # pas utlisé sauf si on appel smmothedeges (mais trop long/ gain qualité)
        factorChain = 10  # 10 OK
        factorApprox = 0.01  # pas utlisé sauf si approxShape (donne style assez rectanglulaire)
        SavingPath += "HQ_%.0f_%s_%s" % (factorResize, name, endName)
        newImage = railImage.createMonoColorImageFactor(factorResize, toulouse.getRGBBackgroundColor())
        colorRail = toulouse.getRGBRailTrackColor()
        for S in railShapes:
            S.chaikins_corner_cutting(factorChain)
            S.resizeShape(factorResize)
            newImage.colorShape(S.getTransformedShape(), colorRail)
        for S in natureShapes:
            colorNature = toulouse.pickNatureColor(FakeRandobj)
            S.chaikins_corner_cutting(factorChain)
            S.resizeShape(factorResize)
            newImage.colorShape(S.getTransformedShape(), colorNature)
        colorAirport = toulouse.getRGBAirportColor()
        for S in aeroportShapes:
            S.chaikins_corner_cutting(factorChain)
            S.resizeShape(factorResize)
            newImage.colorShape(S.getTransformedShape(), colorAirport)
        for S in manMadeShapes:
            colorManMade = toulouse.pickManMadeColor(FakeRandobj)
            S.chaikins_corner_cutting(factorChain)
            S.resizeShape(factorResize)
            # S.approxShape(0.005)
            newImage.colorShape(S.getTransformedShape(), colorManMade)
        colorWater = toulouse.getRGBWaterColor()
        for S in waterShapes:
            S.chaikins_corner_cutting(factorChain)
            S.resizeShape(factorResize)
            newImage.colorShape(S.getTransformedShape(), colorWater)
    newImage.save2PDF(SavingPath+".pdf")
    # newImage.saveImage(SavingPath+".png")

    # pdf_bytes = img2pdf.convert(SavingPath+".png", allow_oversized=True)
    # file = open(SavingPath+".pdf", "wb")
    # file.write(pdf_bytes)
    # file.close()
    print("Completed")
    return None


if __name__ == "__main__":
    key = "test"
    Main(key, True)
