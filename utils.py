import random


def GetRandomColor():
    R = random.randrange(0, 255)
    G = random.randrange(0, 255)
    B = random.randrange(0, 255)
    return (R, G, B)
