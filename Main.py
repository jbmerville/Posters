from Image import Image
from Colors import Toulouse


def Main():
    # originalImage = Image("Shapes.jpg")
    originalImage = Image("/Users/matthieumerville/Desktop/Art/P1.png")

    toulouse = Toulouse()
    newImage = originalImage.createMonoColorImage(toulouse.getRGBBackgroundColor())
    newImage.displayV2(size=7)
    print(str(len(originalImage.getShapes()))+" Shapes")
    for shape in originalImage.getShapes():
        color = toulouse.pickColor()
        newImage.colorShape(shape, color)
    newImage.displayV2(size=7)


if __name__ == "__main__":
    Main()
