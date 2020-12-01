import random
from PIL import ImageColor
from enum import Enum
from utils import FakeRand


class Color:
    def pickManMadeColor(self, FakeRandobj: FakeRand):
        return self.getRGBManMadeColors()[FakeRandobj.Randint(len(self.manMadeColors)-1)]

    def pickNatureColor(self, FakeRandobj: FakeRand):
        return self.getRGBNatureColors()[FakeRandobj.Randint(len(self.natureColors)-1)]

    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


class InputColorType(Enum):
    WATER = 1
    FOREST = 2
    ROAD = 3
    OTHER = 4
    MANMADE = 5
    HOUSE = 6
    AIRPORT = 7
    HORSETRACK = 8
    RAIL = 9


class Paris(Color):
    def __init__(self):
        self.backgroundColor = "#fcecd4"
        self.manMadeColors = ["#212529", "#343a40", "#495057", "#6c757d"]
        self.natureColors = ["#a3b18a", "#588157", "#3a5a40", "#344e41"]
        self.water = "#75aaff"

    def getRGBManMadeColors(self):
        return [ImageColor.getcolor(color, "RGB") for color in self.manMadeColors]

    def getRGBNatureColors(self):
        return [ImageColor.getcolor(color, "RGB") for color in self.natureColors]

    def getRGBWaterColor(self):
        return ImageColor.getcolor(self.water, "RGB")

    def getRGBBackgroundColor(self):
        return ImageColor.getcolor(self.backgroundColor, "RGB")


class Toulouse(Color):
    def __init__(self):
        self.backgroundColor = "#d5ecfc"  # fcecd4" "#D5ECFC"
        self.manMadeColors = ["#edab95", "#e26d66", "#2e2d35", "#356069"]
        self.natureColors = ["#a3b18a", "#588157", "#3a5a40", "#344e41"]
        self.water = "#4dacb2"
        self.railtrack = "#808080"
        self.airport = "#C0C0C0"

    def getRGBManMadeColors(self):
        return [ImageColor.getcolor(color, "RGB") for color in self.manMadeColors]

    def getRGBNatureColors(self):
        return [ImageColor.getcolor(color, "RGB") for color in self.natureColors]

    def getRGBWaterColor(self):
        return ImageColor.getcolor(self.water, "RGB")

    def getRGBRailTrackColor(self):
        return ImageColor.getcolor(self.railtrack, "RGB")

    def getRGBAirportColor(self):
        return ImageColor.getcolor(self.airport, "RGB")

    def getRGBBackgroundColor(self):
        return ImageColor.getcolor(self.backgroundColor, "RGB")
