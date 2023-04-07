import numpy as np
import matplotlib.pyplot as plt

def main(world_name, numberOfAgents):
    performanceMatrix = []
    worldPath = "original_world\world_{}.png".format(world_name)

    performancePath =  "performance\{}_world\_{}_agents\performance_trial_{}.txt".format(world_name, numberOfAgents, 1)
    performance = np.loadtxt(performancePath)
    iteration = performance.shape[0]

    maxIterations = iteration
    for i in range(1, 6):
        performancePath =  "performance\{}_world\_{}_agents\performance_trial_{}.txt".format(world_name, numberOfAgents, i)
        performance = np.loadtxt(performancePath)
        performanceMatrix.append(performance)
        if performance.shape[0] > maxIterations:
            maxIterations = performance.shape[0]

    for i in range(5):
        if performanceMatrix[i].shape[0] != maxIterations:
            added_array = np.ones(maxIterations - performanceMatrix[i].shape[0]) * max(performanceMatrix[i])
            performanceMatrix[i] = np.concatenate((performanceMatrix[i], added_array))
    average_performance = np.zeros(maxIterations)
    for i in range(5):
        average_performance += performanceMatrix[i]
    average_performance = average_performance / 5
    performanceMatrix.append(average_performance)
    performanceMatrix = np.array(performanceMatrix)

    numberOfIterations = np.arange(1, maxIterations + 1, dtype= np.int16)
    title = 'Performance of vector model with {} world with {} iterations'.format(world_name, maxIterations)
    plt.figure()
    plt.plot(performanceMatrix[0], numberOfIterations, 'r-',  linewidth=2)
    plt.plot(performanceMatrix[1], numberOfIterations, 'g-',  linewidth=2)
    plt.plot(performanceMatrix[2], numberOfIterations, 'b-',  linewidth=2)
    plt.plot(performanceMatrix[3], numberOfIterations, 'c-',  linewidth=2)
    plt.plot(performanceMatrix[4], numberOfIterations, 'm-',  linewidth=2)
    plt.plot(performanceMatrix[5], numberOfIterations, 'y-',  linewidth=2)
    plt.xlabel('Coverage (%)')
    plt.ylabel('Iterations')
    plt.title(title)
    plt.legend(['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Average'])
    plt.grid()
    graph_path = "graph\{}_world\_{}_agents\performance_total_{}_world_{}_agents_subplot.png".format(world_name, numberOfAgents, world_name, numberOfAgents)
    plt.savefig(graph_path)
    plt.show()

if __name__ == '__main__':
    for i in range(20, 41, 10):
        main(world_name="obstacle1", numberOfAgents= i)