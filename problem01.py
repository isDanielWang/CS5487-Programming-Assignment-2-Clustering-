'''
@Auther: Shixiang WANG
@EID: sxwang6
@Date: 10/03/2022

@Discription:
This file tests and plots three cluster algorithms: K-means, Gaussian mixture models and Mean-shift algorithm which are implented in 'cluster'. 
'''
# Set up
import processing
import numpy as np
import cluster
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_original():
    """Plot the groud truth cluster.
    """    
    data_names = ['A', 'B', 'C']

    for name in data_names:
        data, labels = processing.data_loader(name)
        colors = cm.rainbow(np.linspace(0, 1, 4))
        cluster_counter = np.zeros(4)
        cluster_centers = np.zeros((4, 2))

        for i in range(data.shape[0]):
            cluster_centers[int(labels[i][0])-1, :] +=  data[i, :]
            cluster_counter[int(labels[i][0])-1] += 1 
            plt.plot(data[i, 0], data[i, 1], 'ko', color=colors[int(labels[i][0]) - 1], markersize=3)
        
        for k in range(cluster_centers.shape[0]):
            cluster_centers[k, :] /= cluster_counter[k]
            plt.plot(cluster_centers[k, 0], cluster_centers[k, 1], '*', color=colors[k], markersize=12, label=f'Cluster:{k+1}')
        
        plt.title(f'Ground truth, Dataset: {name}')
        plt.xlabel('X1'),  plt.ylabel('X2'), plt.grid(True), plt.legend(fontsize=7)
        plt.show()


def plot_kmeans(iterations = 1000):
    """plot the outcome of k-means

    Parameters
    ----------
    iterations, optional
       controls the iteration times, by default 1000
    """    
    data_names = ['A', 'B', 'C']
    for name in data_names:

        data, _ = processing.data_loader(name)
        initial_centers = np.random.uniform(-15, 15, 4 * 2).reshape((4, 2))
        curr_centers, Z = cluster.kmeans(data, initial_centers, iterations)
        print(f'Algorithm: Kmeans, dataset is {name},  final centers: {curr_centers}')
        colors = cm.rainbow(np.linspace(0, 1, curr_centers.shape[0]))

        for i in range(data.shape[0]):

            plt.plot(data[i, 0], data[i, 1], 'ko', color=colors[np.argmax(Z[i, :])], markersize=3)
        
        for k in range(curr_centers.shape[0]):
            
            plt.plot(curr_centers[k, 0], curr_centers[k, 1], '*', color=colors[k], markersize=12, label=f'Cluster:{k+1}')
        
        plt.title(f'Algorithm: Kmeans, Iterations: {iterations}, Dataset: {name}')
        plt.xlabel('X1'),  plt.ylabel('X2'), plt.grid(True), plt.legend(fontsize=7)
        plt.show()

def plot_GMM(iterations = 200):
    """plot the outcome of EM-GMM

    Parameters
    ----------
    iterations, optional
        controls the iteration times, by default 200
    """    
    data_names = ['A', 'B', 'C']
    for name in data_names:

        data, _ = processing.data_loader(name)
        initial_pi = np.array([0.25, 0.25, 0.25, 0.25])
        initial_mean = np.random.rand(4, data.shape[1])
        initial_covariance =np.array([np.eye(data.shape[1])] * 4)
        pi, mean, _, Z = cluster.EM_GMM(data, initial_pi, initial_mean,initial_covariance, iterations)
        print(f'Algorithm: EM-GMM, Dataset: {name}, final centers: {mean}, fianl priors: {pi}')
        colors = cm.rainbow(np.linspace(0, 1, mean.shape[0]))
        
        for i in range(data.shape[0]):

            plt.plot(data[i, 0], data[i, 1], 'ko', color=colors[np.argmax(Z[i, :])], markersize=4)
        
        for k in range(mean.shape[0]):

            plt.plot(mean[k, 0], mean[k, 1], '*', color=colors[k], markersize=12, label=f'Cluster:{k+1}')

        plt.title(f'Algorithm: EM-GMM, Iterations: {iterations}, Dataset: {name}')
        plt.xlabel('X1'),  plt.ylabel('X2'), plt.grid(True), plt.legend(fontsize=7)
        plt.show()

def plot_mean_shift(bandwidth = 3, error = 0.0001,  threshold = 3):
    """_plot the outcome of Mean Shift

    Parameters
    ----------
    bandwidth, optional
        width of the hypercube, by default 3
    error, optional
        control the iteration of the gradient ascent, by default 0.0001
    threshold, optional
        control distance of different peaks, by default 3
    """    
    data_names = ['A', 'B', 'C']
    for name in data_names:

        data, _ = processing.data_loader(name)
        peaks, Z = cluster.mean_shift(data, bandwidth, error, threshold)
        print(f'Algorithm: Mean-shift, Dataset: {name}, final models: {peaks}')
        colors = cm.rainbow(np.linspace(0, 1, len(peaks)))
        
        for i in range(data.shape[0]):
            
            plt.plot(data[i, 0], data[i, 1], 'ko', color=colors[int(Z[i])], markersize=4)
        
        for k in range(len(peaks)):
            
            plt.plot(peaks[k][0], peaks[k][1], '*', color=colors[k], markersize=12, label=f'Cluster:{k+1}')
        
        plt.title(f'Algorithm: Mean-shift, Bandwidth: {bandwidth}, Dataset: {name}')
        plt.xlabel('X1'),  plt.ylabel('X2'), plt.grid(True), plt.legend(fontsize=7)
        plt.show()

if __name__ == '__main__':
    # plot_original()

    # plot_kmeans(iterations = 100)

    # plot_GMM(iterations = 300)

    plot_mean_shift(bandwidth = 3, error = 0.0001,  threshold = 3)