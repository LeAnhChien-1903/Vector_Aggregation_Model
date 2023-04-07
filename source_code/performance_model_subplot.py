import numpy as np
import matplotlib.pyplot as plt
def main(world_name, numberOfAgents, percentage):
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
    figure, axs = plt.subplots(2, 3)
    title = 'Performance of vector model with {} world with {} iterations'.format(world_name, maxIterations)
    figure.suptitle(title)
    
    axs[0, 0].plot( performanceMatrix[0], numberOfIterations, 'r-',  linewidth=2)
    axs[0, 0].set_title('Trial 1')
    axs[0, 0].grid()
    axs[0, 0].set_ylim(0, maxIterations)
    performance_than = np.where(performanceMatrix[0] > percentage)
    position_than = performance_than[0][0]
    axs[0, 0].plot((percentage, percentage), (1, position_than), 'k--', linewidth=1)
    axs[0, 0].plot((1, percentage), (position_than, position_than), 'k--', linewidth=1)
    axs[0, 0].text(50, position_than, str(position_than), fontsize= 10, color='k')
    axs[0, 0].text(percentage, 0, '{}%'.format(percentage), fontsize= 10, color='k')

    axs[0, 1].plot( performanceMatrix[1], numberOfIterations, 'g-',  linewidth=2)
    axs[0, 1].set_title('Trial 2')
    axs[0, 1].grid()
    axs[0, 1].set_ylim(0, maxIterations)
    performance_than = np.where(performanceMatrix[1] > percentage)
    position_than = performance_than[0][0] + 1
    axs[0, 1].plot((percentage, percentage), (1, position_than), 'k--', linewidth=1)
    axs[0, 1].plot((1, percentage), (position_than, position_than), 'k--', linewidth=1)
    axs[0, 1].text(50, position_than, str(position_than), fontsize= 10, color='k')
    axs[0, 1].text(percentage, 0,'{}%'.format(percentage), fontsize= 10, color='k')

    axs[0, 2].plot( performanceMatrix[2], numberOfIterations, 'b-',  linewidth=2)
    axs[0, 2].set_title('Trial 3')
    axs[0, 2].grid()
    axs[0, 2].set_ylim(0, maxIterations)
    performance_than = np.where(performanceMatrix[2] > percentage)
    position_than = performance_than[0][0] + 1
    axs[0, 2].plot((percentage, percentage), (1, position_than), 'k--', linewidth=1)
    axs[0, 2].plot((1, percentage), (position_than, position_than), 'k--', linewidth=1)
    axs[0, 2].text(50, position_than, str(position_than), fontsize= 10, color='k')
    axs[0, 2].text(percentage, 0,'{}%'.format(percentage), fontsize= 10, color='k')
    # axs[0, 3].imshow(world)
    # axs[0, 3].set_title('World')

    axs[1, 0].plot( performanceMatrix[3], numberOfIterations, 'c-',  linewidth=2)
    axs[1, 0].set_title('Trial 4')
    axs[1, 0].grid()
    axs[1, 0].set_ylim(0, maxIterations)
    performance_than = np.where(performanceMatrix[3] > percentage)
    position_than = performance_than[0][0] + 1
    axs[1, 0].plot((percentage, percentage), (1, position_than), 'k--', linewidth=1)
    axs[1, 0].plot((1, percentage), (position_than, position_than), 'k--', linewidth=1)
    axs[1, 0].text(50, position_than, str(position_than), fontsize= 10, color='k')
    axs[1, 0].text(percentage, 0, '{}%'.format(percentage), fontsize= 10, color='k')

    axs[1, 1].plot(performanceMatrix[4], numberOfIterations, 'm-',  linewidth=2)
    axs[1, 1].set_title('Trial 5')
    axs[1, 1].grid()
    axs[1, 1].set_ylim(0, maxIterations)
    performance_than = np.where(performanceMatrix[4] > percentage)
    position_than = performance_than[0][0] + 1
    axs[1, 1].plot((percentage, percentage), (1, position_than), 'k--', linewidth=1)
    axs[1, 1].plot((1, percentage), (position_than, position_than), 'k--', linewidth=1)
    axs[1, 1].text(50, position_than, str(position_than), fontsize= 10, color='k')
    axs[1, 1].text(percentage, 0, '{}%'.format(percentage), fontsize= 10, color='k')

    axs[1, 2].plot( performanceMatrix[5],numberOfIterations, 'y-',  linewidth=2)
    axs[1, 2].set_title('Average')
    axs[1, 2].grid()
    axs[1, 2].set_ylim(0, maxIterations)
    performance_than = np.where(performanceMatrix[5] > percentage)
    position_than = performance_than[0][0] + 1
    axs[1, 2].plot((percentage, percentage), (1, position_than), 'k--', linewidth=1)
    axs[1, 2].plot((1, percentage), (position_than, position_than), 'k--', linewidth=1)
    axs[1, 2].text(50, position_than, str(position_than), fontsize= 10, color='k')
    axs[1, 2].text(percentage, 0, '{}%'.format(percentage), fontsize= 10, color='k')
    # axs[1, 3].imshow(world_covered)
    # axs[1, 3].set_title('World Covered')

    for ax in axs.flat:
        ax.set(xlabel='Performance (%)', ylabel='Iteration')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    graph_path = "graph\{}_world\_{}_agents\performance_{}_world_{}_agents_subplot.png".format(world_name, numberOfAgents, world_name, numberOfAgents)
    plt.savefig(graph_path)
    plt.show()

if __name__ == '__main__':
    main(world_name="obstacle4", numberOfAgents= 40, percentage= 97)