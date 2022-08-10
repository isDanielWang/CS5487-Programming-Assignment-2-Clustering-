'''
@Auther: Shixiang WANG
@EID: sxwang6
@Date: 08/03/2022

@Discription:
This file implements three cluster algorithms: K-means, Gaussian mixture models and Mean-shift algorithm. 
'''

#  Set up
import numpy as np
from scipy.stats import multivariate_normal

'''
****** 1. Start of the Kmeans Algorithm ******
'''

def kmeans_cluster_assignment(data, current_centers):
    """Assign point to its nearest center (Hard assignment)

    Parameters
    ----------
    data
        input data
    current_centers
        ceneters berfor this assignment

    Returns
    -------
        A matrix contains information about the result of the center assignment. 
    """    
    Z = np.zeros((data.shape[0],current_centers.shape[0]))
    for i in range(data.shape[0]):
        distance = float('inf')
        j = 0
        for k in range(current_centers.shape[0]):
            if np.linalg.norm(np.array(data[i,:]) - np.array(current_centers[k, :])) < distance:
                distance = np.linalg.norm(np.array(data[i,:]) - np.array(current_centers[k, :]))
                j = k
        Z[i, j] = 1
    return Z

def kmeans_cenetr_estimation(data, Z):
    """Calculate the location of the centers after once assignment

    Parameters
    ----------
    data
        input data
    Z
        A matrix contains information about the result of the center assignment. 

    Returns
    -------
        updated centers
    """
    new_centers = np.zeros((Z.shape[1], data.shape[1]))
    center_counter = np.zeros(Z.shape[1])

    for i in range(data.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i, j] == 1:
                new_centers[j, :] += data[i, :] 
                center_counter[j] += 1
    
    for j in range(Z.shape[1]):
        if center_counter[j] == 0:
            index = int(np.random.uniform(0, data.shape[1]))
            new_centers[j, :] = data[index, :]
        else:
            new_centers[j, :] =  new_centers[j, :] / center_counter[j]
    
    return new_centers



def kmeans(data, initial_centers, iterations):
    """Train process of K-means 

    Parameters
    ----------
    data
        input data points
    initial_centers
        the random initial locations of centers
    iterations
        iteration times

    Returns
    -------
        centers calculated by kmeans algorithm and  A matrix contains information about the result of the center assignment. 
    """    
    iteration = 0
    while  iteration < iterations:
        iteration += 1
        Z = kmeans_cluster_assignment(data, initial_centers)
        new_centers = kmeans_cenetr_estimation(data, Z)
        initial_centers = new_centers
        if iteration % (iterations // 10) == 0:
            print(f"kmeans iteration: {iteration}/{iterations}")

    return initial_centers, Z

'''
****** 2. Start of the EM algorithm for Gaussian mixture models ******
'''

def Expectation_step(data, pi, mean, covariance):
    """The E-step of the EM alogrithm(soft assignment)

    Parameters
    ----------
    data
        input data
    pi
        component priors
    mean
        component means
    covariance
        component covirance

    Returns
    -------
         A matrix contains information about the result of the center assignment. 
    """    
    Z = np.mat(np.zeros((data.shape[0], pi.shape[0])))
    z = np.zeros((data.shape[0], pi.shape[0]))
    for j in range(pi.shape[0]):
        z[:, j] = multivariate_normal.pdf(data, mean[j], covariance[j])
    z = np.mat(z)
    for k in range(pi.shape[0]):
        Z[:, k] = pi[k] * z[:, k]
    for i in range(data.shape[0]):
        Z[i, :] =  Z[i, :] / np.sum(Z[i, :])
    return Z

def Calculate_LL(data, pi, mean, covariance):
    """Calculate the log likelihood

    Parameters
    ----------
    data
        input points
    pi
        component priors
    mean
        component means 
    covariance
        component variance

    Returns
    -------
        log likelihood of this iteration
    """    
    Q = 0
    for i in range(data.shape[0]):
        l = 0
        for k in range(pi.shape[0]):
            l += pi[k] * multivariate_normal.pdf(data[i, :], mean[k, :], covariance[k, :])
        Q += np.log(l)
    return Q

def Maximum_step(data, Z):
    """The M-step of the EM alogrithm

    Parameters
    ----------
    data
        input points
    Z
        A matrix contains information about the result of the center assignment. 

    Returns
    -------
        the updated componet priors, component mean and variance
    """    
    new_pi, new_mean, new_convariance = np.zeros(Z.shape[1]), np.zeros((Z.shape[1], data.shape[1])), []
    for j in range(Z.shape[1]):
        cluster_counter = np.sum(Z[:, j])
        new_pi[j] = cluster_counter / data.shape[0]
        new_mean[j, :] = np.sum(np.multiply(data, Z[:, j]), axis=0) / cluster_counter 
        variance = np.transpose((data - new_mean[j, :])) * np.multiply((data - new_mean[j, :]), Z[:, j]) / cluster_counter
        new_convariance.append(variance)
    new_convariance = np.array(new_convariance)
    return new_pi, new_mean, new_convariance

def EM_GMM(data, initial_pi, initial_mean, initial_covariance, iterations):
    """trian process of EM algorithm for Gaussian mixture models

    Parameters
    ----------
    data
        input points
    initial_pi
        initial componet priors
    initial_mean
        initial component mean
    initial_covariance
        initial component variance
    iterations
        control when to stop the algorithm 

    Returns
    -------
        the updated componet priors, component mean, component variance and the updated center assignment matrix
    """
    data = np.mat(data)
    # error = float('inf')
    iteration = 0
    while iteration < iterations:
        iteration += 1
        # E-step
        Z = Expectation_step(data, initial_pi, initial_mean, initial_covariance)
        # # Calculate LL
        # q = Calculate_LL(data, initial_pi, initial_mean, initial_covariance)
        # M-step
        initial_pi, initial_mean, initial_covariance = Maximum_step(data, Z)
        # # reassignment
        # Z = Expectation_step(data, initial_pi, initial_mean, initial_covariance)
        # update LL
        Q = Calculate_LL(data, initial_pi, initial_mean, initial_covariance)
        # # Calculate erro
        # error = Q - q
        if iteration % (iterations // 10) == 0:
            print(f"EM-GMM iteration: {iteration}/{iterations}, Current LL: {Q}")
    return initial_pi, initial_mean, initial_covariance, Z

'''
****** Start of Mean-shift algorithm ******
'''

def gaussion_kernal(bandwidth, distance):
    """Calculate the output of a Gaussian kernal
    Parameters
    ----------
    bandwidth
        width of hypercube
    distance
        square euclidean distance between to vectore

    Returns
    -------
        output of the Gaussian kernal
    """    
    kernal  = (1 / bandwidth * np.sqrt(2 * np.pi)) * np.exp(-1/2 * (distance / bandwidth) ** 2)
    return kernal

def calculate_neighbours(point, data, bandwidth):
    """Calculate points in the parzen windows

    Parameters
    ----------
    point
        center point
    data
        input data
    bandwidth
        width of hypercube

    Returns
    -------
        points in the hypercube
    """    
    neighbours = []
    for i in range(data.shape[0]):
        distance = np.linalg.norm((np.array(point) - np.array(data[i, :])))
        if distance <= bandwidth:
            neighbours.append(data[i, :])
    neighbours = np.array(neighbours)
    return neighbours

def find_mode(point, data, bandwidth, threshold):
    """Using gradient ascent to move the point uphill

    Parameters
    ----------
    point
        center point
    data
        input data
    bandwidth
        width of hypercube
    threshold
        control iteration times

    Returns
    -------
        the mode of the distribution
    """    
    diff = float('inf')
    while threshold < diff:
        numerator = np.zeros(data.shape[1])
        denumerator = 0
        neighbours = calculate_neighbours(point, data, bandwidth)
        for neighbour in neighbours:
            distance = np.linalg.norm((np.array(point) - np.array(neighbour)))
            numerator += neighbour * gaussion_kernal(bandwidth, distance)
            denumerator += gaussion_kernal(bandwidth, distance)
        peak = numerator / denumerator
        diff = np.linalg.norm((np.array(peak) - np.array(point)))
        point = peak
    return peak

def mean_shift(data, bandwidth, error = 0.0001, threshold = 1):
    """train process of mean shift algorithm

    Parameters
    ----------
    data
        input data
    bandwidth
        width of hypercube
    error, optional
        control the iterations of gradient ascent, by default 0.0001
    threshold, optional
        control distance of different peaks, by default 1

    Returns
    -------
    final peaks of the distribution and the assigments
    """    
    peaks = []
    Z = np.ndarray(data.shape[0])
    for i in range(data.shape[0]):
        if (i+1) % 50 == 0 or (i+1) == data.shape[0]:
            print(f"mean_shift iteration: {i+1}/{data.shape[0]}")
        peak = find_mode(data[i, :], data, bandwidth, error)
        if len(peaks) == 0:
            peaks.append(peak)
            Z[0] = 0
        else:
            distances = []
            for model in peaks:
                distances.append(np.linalg.norm(model - peak))
            if min(distances) < threshold:
                Z[i] = distances.index(min(distances))
            else:
                peaks.append(peak)
                Z[i] = len(peaks) - 1
    return peaks, Z

'''
************************************ END ************************************
'''
