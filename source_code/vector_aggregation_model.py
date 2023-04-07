import numpy as np
import math 
import cv2
import random

from utils import *
# Define used colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
yellow = (0, 255, 255)
pink = (203, 192, 255)
orange = (0, 165, 255)
class FlockBirdAndPheromone():
    def __init__(self, world, numberOfAgents, radiusOfAgents, personalRange, flockRange, 
                stepOfSize = 5, pheromonesDropped = 1, maxStepInCycle = 25):
        self.world = world.copy()
        self.worldPheromone = world.copy()
        self.worldCovered = world.copy()
        self.worldVectors = world.copy()
        self.height = self.world.shape[0]
        self.width = self.world.shape[1]
        self.numberOfAgents = numberOfAgents
        self.radiusOfAgents = radiusOfAgents
        self.personalRange = personalRange
        self.flockRange = flockRange
        self.stepOfSize = stepOfSize
        self.maxStepInCycle = maxStepInCycle
        self.pheromonesDropped = pheromonesDropped
        self.freeWorld = self.findFreeWorld()
        self.performance = 0.0
        self.pheromoneMatrix = np.ones((self.height, self.width))
        self.initialPosition = np.zeros((self.numberOfAgents, 2), dtype=np.int16)
        self.generateInitialPosition()
        self.generatePheromoneMatrix()
        self.agentPositionList = self.initialPosition.copy()
        for i in range(self.numberOfAgents):
            self.dropPheromoneInPosition(self.agentPositionList[i])

    def generateInitialPosition(self):
        for i in range(self.numberOfAgents):
            self.initialPosition[i,0] = random.randint(-50, 50) + int(self.width/2)
            self.initialPosition[i,1] = random.randint(-50, 50) + int(self.height/2)
            while 1:
                self.initialPosition[i,0] = random.randint(-50, 50) + int(self.width/2)
                self.initialPosition[i,1] = random.randint(-50, 50) + int(self.height/2)
                condition = self.world[self.initialPosition[i,1],self.initialPosition[i,0]] == white
                if condition.all() == True:
                    break

    def generatePheromoneMatrix(self):
        for i in range(self.height):
            for j in range(self.width):
                condition1 = self.world[i, j] == white
                if condition1.all() == True:
                    self.pheromoneMatrix[i, j] = 0.0
                    continue
                condition2 = self.world[i, j] == black
                if condition2.all() == True:
                    self.pheromoneMatrix[i, j] = float('inf')
                    continue

    def main(self):
        totalVectorList = []
        for agent in range(self.numberOfAgents):
            agentPosition = self.agentPositionList[agent].copy()
            vectorList = []
            # Create vector list include interaction flock
            for other in range(self.numberOfAgents):
                if other == agent: continue
                other_position = self.agentPositionList[other].copy()
                vector = self.computeVectorFromTwoAgents(agentPosition, other_position, self.stepOfSize)
                vectorList.append(vector)
            
            # Compute total flock interaction
            flockVector = sumOfListVectors(vectorList) # Flock vector 
            
            # Compute attract vector of min pheromone position
            if flockVector[0] != 0.0:
                attractVector = self.findVectorToMinPheromone(agentPosition, 5, flockVector[0] * 2)
                totalVector = sumOfTwoVectors(flockVector, attractVector)
            else:
                totalVector = self.findVectorToMinPheromone(agentPosition, 5, random.randint(5, 10))
    
            # Limit distance to new position
            if totalVector[0] > self.maxStepInCycle: totalVector[0] = self.maxStepInCycle
            if totalVector[0] ==  0: totalVector[0] = random.randint(1, self.stepOfSize)
            # Compute new position from total vector and current position
            newPositionX = agentPosition[0] + totalVector[0] * math.cos(totalVector[1])
            newPositionY = agentPosition[1] + totalVector[0] * math.sin(totalVector[1])
            newPosition = np.array([newPositionX, newPositionY], dtype= np.int16)
            
            # Check new position is allowed
            if self.positionIsAllowed(agentPosition, newPosition):
                self.agentPositionList[agent] = newPosition.copy()
            else:
                while True:
                    totalVector[0] = totalVector[0] * random.random()
                    if totalVector[0] != 0: break
                totalVector[1] = totalVector[0] + math.pi # Inverse total vector

                newPositionX = agentPosition[0] + totalVector[0] * math.cos(totalVector[1])
                newPositionY = agentPosition[1] + totalVector[0] * math.sin(totalVector[1])
                newPosition = np.array([newPositionX, newPositionY], dtype= np.int16)
                #  Check new position is allowed again
                if self.positionIsAllowed(agentPosition, newPosition):
                    self.agentPositionList[agent] = newPosition.copy()

            totalVectorList.append(totalVector)
        
        # Update world covered
        self.worldPheromone = drawCircles(self.worldPheromone, self.agentPositionList, self.radiusOfAgents, color= green, thickness= -1)
        self.worldCovered = drawCircles(self.worldPheromone.copy(), self.agentPositionList, self.radiusOfAgents, color=blue, thickness= -1)
        self.worldCovered = drawCircles(self.worldCovered, self.agentPositionList, self.flockRange, color=orange,thickness= 1)
        
        # Update pheromone matrix and draw vectors
        for i in range(self.numberOfAgents):
            self.dropPheromoneInPosition(self.agentPositionList[i])
            self.worldCovered = drawVector(self.worldCovered, self.agentPositionList[i], totalVectorList[i], color= red, thickness= 1)
        # Update performance
        self.computePerformance()

    def dropPheromoneInPosition(self, position):
        x = np.arange(position[0] - self.radiusOfAgents, position[0] + self.radiusOfAgents + 1, 1)
        y = np.arange(position[1] - self.radiusOfAgents, position[1] + self.radiusOfAgents + 1, 1)
        for i in y:
            for j in x:
                self.pheromoneMatrix[i, j] += self.pheromonesDropped

    def positionIsAllowed(self, position, new_position):
        if self.positionIsInObstacleAndOutMap(new_position) == True:
            return False
        else:
            if self.connectIsAllowed(position, new_position) == False:
                return False
            else:
                distance = computeDistance(position, new_position)
                if distance > self.maxStepInCycle:
                    return False
        
        return True

    def positionIsInObstacleAndOutMap(self, position):
        x = np.arange(position[0] - self.radiusOfAgents, position[0] + self.radiusOfAgents + 1, 1)
        y = np.arange(position[1] - self.radiusOfAgents, position[1] + self.radiusOfAgents + 1, 1)
        for i in range(x.shape[0]):
            if x[i] >= self.width or x[i] < 0:             
                return True
            for j in range(y.shape[0]):
                if y[j] >= self.height or y[j] < 0:
                    return True
                if self.pheromoneMatrix[y[j], x[i]] == float('inf'):
                    return True
        return False
    def connectIsAllowed(self, pointA, pointB):
        xA = pointA[0]
        yA = pointA[1]
        xB = pointB[0]
        yB = pointB[1]
        if xA != xB  and yA != yB: 
            if xA <= xB:
                x = np.arange(xA, xB + 1, 1)
            else: 
                x = np.arange(xB, xA + 1, 1)
            y = (x - xA) * (yB - yA) / (xB - xA) + yA
        elif xA == xB and yA != yB:
            if yA <= yB:
                y = np.arange(yA, yB + 1, 1)
            else: 
                y = np.arange(yB, yA + 1, 1)
            x = np.ones(y.shape[0]) * xA
        elif yA == yB and xA != xB:
            if xA <= xB:
                x = np.arange(xA, xB + 1, 1)
            else: 
                x = np.arange(xB, xA + 1, 1)
            y = np.ones(x.shape[0]) * yA
        else:
            return True
        for i in range(x.shape[0]):
            if x[i] >= self.width or x[i] < 0:             
                return False 
            for j in range(y.shape[0]):
                if y[j] >= self.height or y[j] < 0:
                    return False
                if self.pheromoneMatrix[int(y[j]), int(x[i])] == float('inf'):
                            return False
        return True
    def findFreeWorld(self):
        freeWorld = getFreeWorld(self.world.copy(), self.height, self.width)
        
        return freeWorld

    def computePerformance(self):
        subtract = cv2.subtract(self.world.copy(), self.worldCovered)
        worldCovered = getFreeWorld(subtract, self.height, self.width)
        self.performance = worldCovered / self.freeWorld * 100

    def computeVectorFromTwoAgents(self, agent, otherAgent, magnitude):
        distance = computeDistance(agent, otherAgent)
        if distance <= self.personalRange:
            Ax, Ay = otherAgent
            Bx, By = agent
            theta_rad = math.atan2(By - Ay, Bx - Ax + 0.000000001)
        elif self.personalRange < distance <= self.flockRange:
            Ax, Ay = agent
            Bx, By = otherAgent
            theta_rad = math.atan2(By - Ay, Bx - Ax + 0.000000001)     
        else: 
            magnitude = 0.0
            theta_rad = 0.0

        return np.array((magnitude, theta_rad))
    def findVectorToMinPheromone(self, position, range, magnitude):
        subMatrix = self.extractSquareSubMatrix(position, range)
        minIndex = findMinIndex(subMatrix)
        angle = math.atan2(minIndex[1], minIndex[0])
        i = 0
        while i < 5:
            i += 1 
            new_point = np.zeros(2, dtype= np.int16)
            new_point[0] = position[0] + magnitude * math.cos(angle)
            new_point[1] = position[1] + magnitude * math.sin(angle)
            if new_point[0] >= self.width or new_point[1] >= self.height or new_point[0] < 0 or new_point[1] < 0:
                continue 
            if self.world[new_point[1], new_point[0], 0] != 0:
                break
            minIndex = findMinIndex(subMatrix)
            angle = math.atan2(minIndex[1], minIndex[0])

        return np.array((magnitude, angle))
    def extractSquareSubMatrix(self, center, radius):
        # Find the size of square matrix
        size = int(np.ceil(radius*2)) + 1
        sub_matrix = np.zeros((size, size))
        x = np.arange(center[0] - radius, center[0] + radius + 1, 1)
        y = np.arange(center[1] - radius, center[1] + radius + 1, 1)

        for i in range(size):
            for j in range(size):
                if x[i] >= self.width or y[j] >= self.height or x[i] < 0 or y[j] < 0 :
                    sub_matrix[j, i] = float('inf')
                else:
                    sub_matrix[j, i] = self.findSumPheromoneLevel((x[i], y[j]), 5)
        return sub_matrix
    def findAveragePheromoneLevel(self, position):
        pheromoneLevelMatrix = self.pheromoneMatrix[position[1] - self.radiusOfAgents: position[1] + self.radiusOfAgents + 1, position[0] - self.radiusOfAgents : position[0] + self.radiusOfAgents + 1]
        return np.average(pheromoneLevelMatrix)
    def findSumPheromoneLevel(self, position, radius):
        pheromoneLevelMatrix = self.pheromoneMatrix[position[1] - radius: position[1] + radius + 1, position[0] - radius : position[0] + radius + 1]
        return np.sum(pheromoneLevelMatrix)
    @staticmethod
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

    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def computeDistance(position1, position2):
        distance = math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)
        return distance
    @staticmethod
    def drawCircles(world, position, radius, color, thickness):
        new_world = world.copy()
        for i in range(position.shape[0]):
            new_world = cv2.circle(new_world, position[i], radius, color = color, thickness = thickness)
        return new_world
    @staticmethod
    def getFreeWorld(world, height, width):
        channel = world[:, : , 0]
        channel = np.reshape(channel, height * width)
        result = np.where((channel > 0) )
        result = np.array(result[0])
        return result.shape[0]
    @staticmethod
    def drawVector(world, position, vector, color, thickness):
        new_world = world.copy()
        new_position = np.zeros(2, dtype= np.int16)
        new_position[0] = position[0] + vector[0] * math.cos(vector[1])
        new_position[1] = position[1] + vector[0] * math.sin(vector[1])

        new_world = cv2.arrowedLine(new_world, position, new_position, color= color, thickness= thickness)

        return new_world

def main(world_name, numberOfAgents, trials, performance_request, numberOfIterations, mode):
    worldPath =  "original_world\world_" + world_name + ".png"
    world = cv2.imread(worldPath)
    object = FlockBirdAndPheromone(world, numberOfAgents= numberOfAgents, radiusOfAgents= 5, personalRange= 35, flockRange= 55, stepOfSize= 5)
    if mode == 'fix':
        i = 1
        performanceList = []
        while i <= numberOfIterations:
            object.main()
            text = 'Iter %d: %.2f'%(i, object.performance) + '%'
            performanceList.append(object.performance)
            world_covered = cv2.putText(object.worldCovered.copy(), text, (50, 50), fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale= 0.75, color= (0, 0, 255), thickness= 2, lineType=cv2.LINE_AA)
            world_covered_path = 'covered_world\{}_world\_{}_agents\_trial_{}\iteration{}.png'.format(world_name, numberOfAgents, trials, i)
            
            # cv2.imwrite(world_covered_path, world_covered)
            cv2.imshow("Area-coverage", world_covered)
            if cv2.waitKey(1) & ord('q') == 0xFF:
                break
            i += 1
        result = np.array(performanceList)
        performanceFilePath = "performance\{}_world\_{}_agents\performance_trial_{}.txt".format(world_name, numberOfAgents, trials)
        np.savetxt(performanceFilePath, result, fmt='%.2f')
        cv2.destroyAllWindows()
    elif mode == "not_fix":
        i = 1
        performanceList = []
        while object.performance < performance_request:
            object.main()
            text = 'Iter %d: %.2f'%(i, object.performance) + '%'
            performanceList.append(object.performance)
            world_covered = cv2.putText(object.worldCovered.copy(), text, (50, 50), fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale= 0.75, color= (0, 0, 255), thickness= 2, lineType=cv2.LINE_AA)
            world_covered_path = 'covered_world\{}_world\_{}_agents\_trial_{}\iteration{}.png'.format(world_name, numberOfAgents, trials, i)
            
            # cv2.imwrite(world_covered_path, world_covered)
            cv2.imshow("Area-coverage", world_covered)
            if cv2.waitKey(1) & ord('q') == 0xFF:
                break
            i += 1
        result = np.array(performanceList)
        performanceFilePath = "performance\{}_world\_{}_agents\performance_trial_{}.txt".format(world_name, numberOfAgents, trials)
        np.savetxt(performanceFilePath, result, fmt='%.2f')
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main(world_name = "empty", numberOfAgents= 25, trials= 5, performance_request= 96, numberOfIterations= 2000, mode = "not_fix")