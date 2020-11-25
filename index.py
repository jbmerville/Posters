from IMG import IMG
from Process import Process

originalIMG = IMG.getIMG("Shapes.jpg")  # a tester "Shapes.jpg"
IMG.showIMG(originalIMG, 500)
GreyIMG = Process.PrepareIMG(originalIMG)
contours = Process.GetContours(GreyIMG)
print(str(len(contours))+" contours found !")
for index in range(0, len(contours)-1):
    Process.DrawOneContours(originalIMG, contours,
                            index, Process.GetRandomColor())
