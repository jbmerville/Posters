import random


def getRandomColor():
    R = random.randrange(0, 255)
    G = random.randrange(0, 255)
    B = random.randrange(0, 255)
    return (R, G, B)


class FakeRand:
    def __init__(self, key):
        random.seed(key)

    def Randint(self, max):
        return random.randint(0, max)
