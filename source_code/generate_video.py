import cv2
import numpy as np

def main(world_name, numberOfAgents, trial):
    worldPath =  "original_world\world_{}.png".format(world_name)
    world = cv2.imread(worldPath)
    frameSize = (world.shape[1], world.shape[0])
    
    performancePath =  "performance\{}_world\_{}_agents\performance_trial_{}.txt".format(world_name, numberOfAgents, trial)
    performance = np.loadtxt(performancePath)
    iteration = performance.shape[0]

    outPath = '_video\{}_world\_{}_agents\{}_world_{}_trial{}_.avi'.format(world_name, numberOfAgents, world_name, numberOfAgents, trial)
    out = cv2.VideoWriter(outPath,cv2.VideoWriter_fourcc(*'DIVX'), 24, frameSize)

    for i in range(1, iteration + 1):
        filename = 'covered_world\{}_world\_{}_agents\_trial_{}\iteration{}.png'.format(world_name, numberOfAgents, trial, i)
        image = cv2.imread(filename)
        out.write(image)

    out.release()

if __name__ == "__main__":
    for i in range(1, 6):
        main(world_name= 'obstacle4', numberOfAgents= 40, trial= i)
        print("Done {}_{}!".format(40, i))