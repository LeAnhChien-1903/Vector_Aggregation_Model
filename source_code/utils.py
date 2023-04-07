import numpy as np
import math 
import random
import cv2

def findMinIndex(matrix):
    center = int(matrix.shape[0]/2)
    minVal = np.min(matrix)
    minIndex = np.where(matrix == minVal)
    minIndexX = np.array(minIndex[1])
    minIndexY = np.array(minIndex[0])
    if minIndexX.shape[0] == 0:
        return np.array([-center, -center])
    elif minIndexX.shape[0] == 1:
        return np.array([minIndexX[0] - center, minIndexY[0] - center])
    else:
        randomIndex = random.randint(0, minIndexX.shape[0] - 1)
        return np.array([minIndexX[randomIndex] - center, minIndexY[randomIndex] - center])


def sumOfListVectors(vectors):
    Vx = 0.0
    Vy = 0.0
    # Compute x, y parts of vectors
    for i in range(len(vectors)):
        Vx += vectors[i][0] * math.cos(vectors[i][1])
        Vy += vectors[i][0] * math.sin(vectors[i][1])
    # Compute magnitude and angle of sum vector
    magnitude = math.sqrt(Vx ** 2 + Vy ** 2)
    angle = math.atan2(Vy, Vx + 0.0000000001)

    return np.array((magnitude, angle))

def sumOfTwoVectors(vector1, vector2):
    Ax = vector1[0] * math.cos(vector1[1])
    Ay = vector1[0] * math.sin(vector1[1])
    Bx = vector2[0] * math.cos(vector2[1])
    By = vector2[0] * math.sin(vector2[1])

    Cx = Ax + Bx
    Cy = Ay + By

    magnitude = math.sqrt(Cx**2 + Cy**2)
    angle = math.atan2(Cy, Cx)

    return np.array([magnitude, angle])

def computeDistance(position1, position2):
    distance = math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)
    return distance

def drawCircles(world, position, radius, color, thickness):
    new_world = world.copy()
    for i in range(position.shape[0]):
        new_world = cv2.circle(new_world, position[i], radius, color = color, thickness = thickness)
    return new_world

def getFreeWorld(world, height, width):
    channel = world[:, : , 0]
    channel = np.reshape(channel, height * width)
    result = np.where((channel > 0) )
    result = np.array(result[0])
    return result.shape[0]

def drawVector(world, position, vector, color, thickness):
    new_world = world.copy()
    new_position = np.zeros(2, dtype= np.int16)
    new_position[0] = position[0] + vector[0] * math.cos(vector[1])
    new_position[1] = position[1] + vector[0] * math.sin(vector[1])

    new_world = cv2.arrowedLine(new_world, position, new_position, color= color, thickness= thickness)

    return new_world