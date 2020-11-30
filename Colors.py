import random
from PIL import ImageColor


class Color:
    def pickColor(self):
        return self.getRGBColors()[random.randrange(0, len(self.colors), 1)]


class Toulouse(Color):
    def __init__(self):
        self.backgroundColor = "#fcecd4"
        self.colors = ["#e16c66", "#edab96", "#355f68", "#2e2c34"]

    def getRGBColors(self):
        return [ImageColor.getcolor(color, "RGB") for color in self.colors]

    def getRGBBackgroundColor(self):
        return ImageColor.getcolor(self.backgroundColor, "RGB")
