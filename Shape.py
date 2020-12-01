import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from Image import Image
import copy
from Colors import InputColorType
from functools import reduce
import operator
import math


class Shape:

    def __init__(self, cv2contour, shapeType: InputColorType) -> None:
        # single cv2contour (ie shape) and shapeType
        self.originalshape = cv2contour
        self.type = shapeType
        self.transformedshape = copy.deepcopy(cv2contour)

    def shapeType(self):
        return self.shapeType

    def resetTransformedShape(self):
        self.transformedshape = self.originalshape

    def getTransformedShape(self):
        return self.transformedshape.astype(np.int32)

    def hullShape(self):
        return cv2.convexHull(self.originalshape)

    def length(self):
        return cv2.arcLength(self.originalshape, False)

    def Area(self):
        return cv2.contourArea(self.transformedshape)

    def resizeShape(self, factor):
        self.transformedshape[:, :, 0] = self.transformedshape[:, :, 0] * factor
        self.transformedshape[:, :, 1] = self.transformedshape[:, :,  1] * factor

    def getPoints(self):
        pts = []
        for i in self.transformedshape:
            pts.append(i[0])
        pts = np.array(pts)
        return pts

    def pointsToContour(self, points):
        contour = np.array(points).reshape((-1, 1, 2)).astype(np.float32)

        return contour

    def getNumberPoints(self, lenght: int, start=0):
        return self.getPoints()[start:start+lenght]

    def pltPoints(self, points):
        x = list()
        y = list()
        for p in points:
            x.append(p[0])
            y.append((p[1]))
            print(p[0], p[1])

        plt.scatter(x, y)
        plt.plot(x, y)
        plt.show()

    def pltClosedPoints(self, points):
        x = list()
        y = list()

        for p in points:
            x.append(p[0])
            y.append((p[1]))

        x.append(points[0][0])
        y.append(points[0][1])
        plt.scatter(x, y)
        plt.plot(x, y)
        plt.show()

    def pltClosedPointsAndOther(self, points, otherpoints):
        x = list()
        y = list()

        for p in points:
            x.append(p[0])
            y.append((p[1]))
        x.append(points[0][0])
        y.append(points[0][1])
        plt.scatter(x, y)
        plt.plot(x, y)

        for pc in otherpoints:
            x = list()
            y = list()
            p = pc[0]
            c = pc[1]

            x.append(p[0])
            y.append((p[1]))
            plt.scatter(x, y, color=c)
            plt.plot(x, y)

        plt.show()

    def pltShape(self):
        self.pltClosedPoints(self.getPoints())

    def pltShapeAndPts(self, Pts):
        self.pltClosedPointsAndOther(self.getPoints(), Pts)

    def splitShape(self):
        pts = self.getPoints()
        # calcul des vecteurs
        listVect = list()
        for i in range(1, len(pts)):
            listVect.append(self.getVectorDirectorPts(pts[i-1], pts[i]))
        listContour = list()
        subcontour = list()
        subcontour.append(pts[0])
        subcontour.append(pts[1])
        for i in range(0, len(listVect)-1):
            angle = self.getAngleVectors(listVect[i], listVect[i+1])
            if angle > 50:
                listContour.append(np.array(subcontour))
                subcontour = list()
                subcontour.append(pts[i+1])
                subcontour.append(pts[i+2])
            else:
                subcontour.append(pts[i+2])
        listContour.append(np.array(subcontour))
        return listContour

    def getVectorDirectorPts(self, point1, point2):
        vect = [point2[0]-point1[0], point2[1]-point1[1]]
        vect = np.array(vect)

        vect = vect/np.linalg.norm((vect))
        return vect

    def distPts(self, p1, p2):
        return math.sqrt(math.pow(p1[0]-p2[0], 2)+math.pow(p1[1]-p2[1], 2))

    # def getVectorDirector(self, pos: int):
    #     Points = self.getNumberPoints(2, 1)
    #     vect = [Points[1][0]-Points[0][0], Points[1][1]-Points[0][1]]
    #     vect = np.array(vect)
    #     vect = vect/np.linalg.norm((vect))
    #     return vect

    def getAngleVectors(self, vect1, vect2):
        dot_product = np.dot(vect1, vect2)
        angle = np.arccos(dot_product)
        angle = angle/(2*np.pi)*360
        return angle

    def getOffSetContour(self, offwidth, offheight):
        pts = self.getPoints()
        offpts = list()

        for p in pts:
            offpts.append([float(p[0]-offwidth), float(p[1]-offheight)])
        return offpts

    def sampleShape(self, minLength, treshold):
        pts = self.getPoints()

        newPts = list()
        newPts.append(pts[0])

        if len(pts) > minLength:
            i = 0

            while i < len(pts)-1:
                p = pts[i]
                pnext = pts[i+1]
                if self.distPts(p, pnext) < treshold:
                    # on peut supprimer le pts
                    newPts.append(p)
                    i = i+2
                else:
                    newPts.append(p)
                    i = i+1
            newPts.append(pts[len(pts)-1])
            self.transformedshape = self.pointsToContour(newPts)
        else:
            self.transformedshape = self.pointsToContour(pts)

        # print("new lenght %f" % (len(newPts)/len(pts)))
    def orderPtsExtreme(self, index):
        pts = self.getPoints()
        orderPts = list()
        for i in range(len(pts)):
            orderPts.append(pts[(i+index) % len(pts)])
        return np.array(orderPts)

    def minPointsShape(self):
        # on sample la shape
        # self.pltShape()
        self.sampleShape(3, 10)
        # self.pltShape()
        tresholdAngle = 5
        pts = self.getPoints()
        i = 0
        newPts = list()
        newPts.append(pts[0])
        while i < len(pts)-1:
            ptRef = pts[i]
            c = True
            j = min(i+1, len(pts)-1)
            j1 = min(j+1, len(pts)-1)
            while c:
                ptToRemove = pts[j]
                ptNext = pts[j1]
                # calcul des vect
                # self.pltShapeAndPts([[ptToRemove, "green"], [ptRef, "red"], [ptNext, "yellow"]])
                vect1 = self.getVectorDirectorPts(ptRef, ptToRemove)
                vect2 = self.getVectorDirectorPts(ptToRemove, ptNext)
                a = self.getAngleVectors(vect1, vect2)
                if a < tresholdAngle:
                    vect2 = self.getVectorDirectorPts(ptRef, ptNext)
                    a = self.getAngleVectors(vect1, vect2)
                    if a < tresholdAngle:
                        # on peut supprimer le pt
                        j = min(j+1, len(pts)-1)
                        j1 = min(j+1, len(pts)-1)

                    else:
                        # on ne peut pas supprimer le point
                        newPts.append(ptToRemove)
                        i = j
                        c = False

                else:
                    # on ne peut pas supprimer le point
                    newPts.append(ptToRemove)
                    i = j
                    c = False

        self.transformedshape = self.pointsToContour(newPts)
        # self.pltShape()

    def testPropertiesShape(self):
        # self.pltShape()
        if len(self.getPoints()) > 3:
            self.minPointsShape()
            self.pltShape()
            cnt = self.transformedshape
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w*h
            extent = float(area)/rect_area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            print("extent %f" % extent)
            print("solidity %f" % solidity)
            # epsilon = 0.01*cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            # print("approx ")
            # self.transformedshape = approx
            # self.pltShape()

    def solidity(self):
        self.minPointsShape()
        if len(self.getPoints()) > 3:
            cnt = self.transformedshape
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            if solidity < 0.95:
                print(solidity)
                self.pltShape()

    def extent(self):
        self.minPointsShape()
        self.pltShape()
        if len(self.getPoints()) > 1:
            cnt = self.transformedshape
            area = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)

            rect_area = cv2.contourArea(box)
            extent = float(area)/rect_area
            if extent < 0.6:
                print(extent)
                self.pltShape()

    def simplifyShape(self):

        # sampling of the shape to obtain the same shape with less points
        # self.sampleShape(3, 10) # ancien sampling
        self.minPointsShape()
        # getting points
        # self.pltShape()
        pts = self.getPoints()
        # bounding rect to determine the max "size" of the shape to set the dist value
        rect = cv2.minAreaRect(self.transformedshape)
        box = cv2.boxPoints(rect)

        topl = box[0]
        topr = box[1]
        botl = box[3]
        if len(pts) > 5:  # otherwise we don't do nothing
            dist = min(30, 0.8*math.sqrt(math.pow(topl[0]-topr[0], 2)+math.pow(botl[1]-topr[1], 2)),
                       0.8*math.sqrt(math.pow(botl[0]-topr[0], 2)+math.pow(botl[1]-topr[1], 2)))
            dist = 20
            mindist = 15
            maxdist = 30
            stepdist = 2
            A = self.Area()
            t = 10  # 7
            newPts = list()
            i = 0
            theta = -7  # degree
            maxAngle = 30  # 30
            minPtsIndex = np.argmin(pts, axis=0)[0]
            # print(minPtsIndex)

            pts = self.orderPtsExtreme(minPtsIndex)

            while i < len(pts):
                ip1 = (i+1) % len(pts)
                ip2 = (i+2) % len(pts)
                vect1 = self.getVectorDirectorPts(pts[i], pts[ip1])
                vect2 = self.getVectorDirectorPts(pts[ip1], pts[ip2])

                angle = self.getAngleVectors(vect1, vect2)
                if angle > maxAngle:

                    vect = self.getVectorDirectorPts(pts[i], pts[ip1])
                    npx = vect[0]*dist+pts[ip1][0]
                    npy = vect[1]*dist+pts[ip1][1]
                    newPt = [npx, npy]
                    # self.pltShapeAndPts([[newPt, "red"], [pts[ip1], "yellow"]])
                    found = False
                    for j in range(i+3, i+int(0.3*len(pts)),):
                        k = j % len(pts)-1
                        if self.distPts(newPt, pts[k]) < t:

                            newPts.append(pts[i])
                            newPts.append(pts[k])
                            found = True
                            break
                    if found:
                        # print("found")
                        # self.pltShapeAndPts([[pts[ip1], "yellow"], [pts[k], "green"]])
                        i = j

                    else:
                        # on regarde si mathching en décallant d'un angle theta petit le vect directeur

                        theta = (theta/360)*(2*math.pi)
                        vectR = [math.cos(theta)*vect[0]-math.sin(theta)*vect[1],
                                 math.sin(theta)*vect[0]+math.cos(theta)*vect[1]]
                        npx = vectR[0]*dist+pts[ip1][0]
                        npy = vectR[1]*dist+pts[ip1][1]
                        newPt = [npx, npy]

                        # self.pltShapeAndPts([[newPt, "red"], [pts[ip1], "yellow"]])
                        found = False
                        for j in range(i+3, i+int(0.3*len(pts)),):
                            k = j % len(pts)-1

                            if self.distPts(newPt, pts[k]) < t:

                                newPts.append(pts[i])
                                newPts.append(pts[k])
                                found = True

                        if found:
                            # print("found")
                            # self.pltShapeAndPts([[pts[ip1], "yellow"], [pts[k], "green"]])
                            i = j

                        else:

                            theta = -theta
                            vectR = [math.cos(theta)*vect[0]-math.sin(theta)*vect[1],
                                     math.sin(theta)*vect[0]+math.cos(theta)*vect[1]]
                            npx = vectR[0]*dist+pts[ip1][0]
                            npy = vectR[1]*dist+pts[ip1][1]
                            newPt = [npx, npy]

                            # self.pltShapeAndPts([[newPt, "red"], [pts[ip1], "yellow"]])
                            found = False
                            for j in range(i+3, i+int(0.3*len(pts)),):
                                k = j % len(pts)-1

                                if self.distPts(newPt, pts[k]) < t:

                                    newPts.append(pts[i])
                                    newPts.append(pts[k])
                                    found = True
                                    break
                            if found:

                                # print("found")
                                # self.pltShapeAndPts([[pts[i+1], "yellow"], [pts[k], "green"]])
                                i = j

                            else:
                                newPts.append(pts[i])
                                i = i+1
                else:
                    newPts.append(pts[i])
                    i = i+1

            self.transformedshape = self.pointsToContour(newPts)
            # self.pltShape()
            # print("Finished")
            # self.pltShape()

            # if self.Area()/A < 0.9:

            #     self.transformedshape = self.pointsToContour(pts)
         # else:
            # print("nothing")
            # print("Final")
            # self.pltShape()

            # angle detection
            # for i in range(len(pts)-2):
            #     vect1 = self.getVectorDirectorPts(pts[i], pts[i+1])
            #     vect2 = self.getVectorDirectorPts(pts[i+1], pts[i+2])
            #     angle = self.getAngleVectors(vect1, vect2)
            #     print(angle)

            # hull = cv2.convexHull(self.transformedshape)
            # self.transformedshape = hull
            # self.pltShape()

    def simplifyShapeDist(self):

        # sampling of the shape to obtain the same shape with less points
        # self.sampleShape(3, 10) # ancien sampling
        self.minPointsShape()
        # getting points
        # self.pltShape()
        pts = self.getPoints()
        # bounding rect to determine the max "size" of the shape to set the dist value
        rect = cv2.minAreaRect(self.transformedshape)
        box = cv2.boxPoints(rect)

        topl = box[0]
        topr = box[1]
        botl = box[3]
        if len(pts) > 5:  # otherwise we don't do nothing
            dist = min(30, 0.8*math.sqrt(math.pow(topl[0]-topr[0], 2)+math.pow(botl[1]-topr[1], 2)),
                       0.8*math.sqrt(math.pow(botl[0]-topr[0], 2)+math.pow(botl[1]-topr[1], 2)))
            dist = 20
            mindist = 20
            maxdist = 45
            stepdist = 2
            maxArea = 0.95
            A = self.Area()

            t = 7  # 7
            newPts = list()
            i = 0
            theta = -5  # degree
            maxAngle = 30  # 30
            minPtsIndex = np.argmin(pts, axis=0)[0]
            # print(minPtsIndex)

            pts = self.orderPtsExtreme(minPtsIndex)

            while i < len(pts):
                ip1 = (i+1) % len(pts)
                ip2 = (i+2) % len(pts)
                vect1 = self.getVectorDirectorPts(pts[i], pts[ip1])
                vect2 = self.getVectorDirectorPts(pts[ip1], pts[ip2])

                angle = self.getAngleVectors(vect1, vect2)
                if angle > maxAngle:
                    foundDist = False
                    dist = mindist
                    while dist < maxdist and foundDist == False:
                        vect = self.getVectorDirectorPts(pts[i], pts[ip1])
                        npx = vect[0]*dist+pts[ip1][0]
                        npy = vect[1]*dist+pts[ip1][1]
                        newPt = [npx, npy]
                        #self.pltShapeAndPts([[newPt, "red"], [pts[ip1], "yellow"]])
                        found = False

                        for j in range(i+3, i+int(0.4*len(pts)),):
                            k = j % len(pts)-1
                            if self.distPts(newPt, pts[k]) < t:

                                newPts.append(pts[i])
                                newPts.append(pts[k])
                                found = True
                                break
                        if found:
                            # print("found")
                            #self.pltShapeAndPts([[pts[ip1], "yellow"], [pts[k], "green"]])
                            i = j
                            foundDist = True

                        else:
                            # on regarde si mathching en décallant d'un angle theta petit le vect directeur

                            theta = (theta/360)*(2*math.pi)
                            vectR = [math.cos(theta)*vect[0]-math.sin(theta)*vect[1],
                                     math.sin(theta)*vect[0]+math.cos(theta)*vect[1]]
                            npx = vectR[0]*dist+pts[ip1][0]
                            npy = vectR[1]*dist+pts[ip1][1]
                            newPt = [npx, npy]

                            #self.pltShapeAndPts([[newPt, "red"], [pts[ip1], "yellow"]])
                            found = False
                            for j in range(i+3, i+int(0.4*len(pts)),):
                                k = j % len(pts)-1

                                if self.distPts(newPt, pts[k]) < t:

                                    newPts.append(pts[i])
                                    newPts.append(pts[k])
                                    found = True

                            if found:
                                # print("found")
                                #self.pltShapeAndPts([[pts[ip1], "yellow"], [pts[k], "green"]])
                                i = j
                                foundDist = True

                            else:

                                theta = -theta
                                vectR = [math.cos(theta)*vect[0]-math.sin(theta)*vect[1],
                                         math.sin(theta)*vect[0]+math.cos(theta)*vect[1]]
                                npx = vectR[0]*dist+pts[ip1][0]
                                npy = vectR[1]*dist+pts[ip1][1]
                                newPt = [npx, npy]

                                #self.pltShapeAndPts([[newPt, "red"], [pts[ip1], "yellow"]])
                                found = False
                                for j in range(i+3, i+int(0.4*len(pts)),):
                                    k = j % len(pts)-1

                                    if self.distPts(newPt, pts[k]) < t:

                                        newPts.append(pts[i])
                                        newPts.append(pts[k])
                                        found = True
                                        break
                                if found:

                                    # print("found")
                                    #self.pltShapeAndPts([[pts[i+1], "yellow"], [pts[k], "green"]])
                                    i = j
                                    foundDist = True

                                else:
                                    # newPts.append(pts[i])
                                    #i = i+1
                                    dist = dist+stepdist

                    if foundDist == False:
                        newPts.append(pts[i])
                        i = i+1

                # end loop
                else:
                    newPts.append(pts[i])
                    i = i+1

            self.transformedshape = self.pointsToContour(newPts)
            if A == 0:
                self.transformedshape = self.pointsToContour(pts)
            elif self.Area()/A < maxArea:
                self.transformedshape = self.pointsToContour(pts)
            # self.pltShape()

    def approxShape(self, perc):
        epsilon = perc*cv2.arcLength(self.transformedshape, True)

        approxS = cv2.approxPolyDP(self.transformedshape, epsilon, True)
        self.transformedshape = approxS

    def smoothedges(self, size):
        # a améliorer (dimensioner l'image pour appliquer la  transformation par rapport à la taille de la forme)

        if self.Area() > 1:

            cnt = self.transformedshape
            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

            maxY = bottommost[1]
            minY = topmost[1]
            maxX = rightmost[0]
            minX = leftmost[0]

            rect = cv2.boundingRect(self.transformedshape)

            offwidth = minX-(size+4)
            offheight = minY-(size+4)

            widthImg = maxX-minX+2*(size+4)
            heightImg = maxY-minY+2*(size+4)

            blank_image = np.zeros((int(heightImg)+1, int(widthImg)+1, 3), np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            color = (0, 0, 0)
            monoImage = Image("", blank_image)
            monoImage.revertImage()

            offsetContour = self.getOffSetContour(offwidth, offheight)

            offsetContour = self.pointsToContour(offsetContour).astype(np.int32)

            monoImage.colorShape(offsetContour, color)

            monoImage.img = cv2.morphologyEx(monoImage.img, cv2.MORPH_OPEN, kernel, iterations=4)

            smoothedShapes = monoImage.getShapes()

            if len(smoothedShapes) > 0:
                self.transformedshape = smoothedShapes[0]
                newPts = self.getOffSetContour(-offwidth, -offheight)

                self.transformedshape = self.pointsToContour(newPts).astype(np.int32)

    def chaikins_corner_cutting(self, refinements=5):
        coords = np.array(self.getPoints())

        for _ in range(refinements):
            L = coords.repeat(2, axis=0)
            R = np.empty_like(L)
            R[0] = L[0]
            R[2:: 2] = L[1: -1: 2]
            R[1: -1: 2] = L[2:: 2]
            R[-1] = L[-1]
            coords = L * 0.75 + R * 0.25

        self.transformedshape = self.pointsToContour(coords)

    def smoothedgescubic(self):
        # cubic interpolation (utile ?)
        try:
            pts = self.getPoints()
            tck, u = splprep(pts.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)

            plt.plot(pts[:, 0], pts[:, 1], 'ro')
            plt.plot(x_new, y_new, 'b--')
            plt.show()
            l = list()
            for i in range(0, len(x_new)):
                l.append([[x_new[i], y_new[i]]])
            # self.transformedshape = np.array(l).reshape((-1, 1, 2)).astype(np.int32)
            self.transformedshape = np.array(l).astype(np.int32)
        except:
            return None
