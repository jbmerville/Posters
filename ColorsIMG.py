import random


class Color:
    class ToulouseStyle:

        def __init__(self):
            self.BackgroundColor = (252, 236, 212)

        def ColorList(self):
            ColorL = list()
            C1 = (225, 108, 102)
            C2 = (237, 171, 150)
            C3 = (53, 95, 104)
            C4 = (46, 44, 52)
            ColorL.append(C1)
            ColorL.append(C2)
            ColorL.append(C3)
            ColorL.append(C4)
            return ColorL

    def PickColor(Colorlist):
        print(Colorlist)
        return Colorlist[random.randrange(0, len(Colorlist), 1)]
