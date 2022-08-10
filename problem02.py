'''
@Auther: Shixiang WANG
@EID: sxwang6
@Date: 12/03/2022

@Discription:
This file solves a real world clustering problem: image segmentation using the three cluster algorithms implemented in 'cluster.py'.
'''

# Set up
import pa2
import cluster
import scipy.cluster.vq as vq
import PIL.Image as Image
import numpy as np
import pylab as pl

def load_img(img_name):
    """Load image by its name

    Parameters
    ----------
    img_name
        image name

    Returns
    -------
        image
    """    
    img = Image.open('data/PA2-images/'+ img_name + '.jpg')
    return img


def plot(algorithm, iterations = 200, bandwidth = 0.5, clusters = 3, normalization = False):
    """Plot the outcome for problem 2

    Parameters
    ----------
    algorithm
        name of the using algorithm
    iterations, optional
        iteration times, by default 200
    bandwidth, optional
        the bandwidth value of mean-shift, by default 0.5
    clusters, optional
        the number of clusters, by default 3
    normalization, optional
        normalize the input data, by default False
    """    
    images = ['12003', '299086', '56028', '62096']
    for image in images:
        img = load_img(image)
        pl.subplot(len(images),3,3*images.index(image)+1)
        # pl.axis('off')
        pl.imshow(img)


        # extract features from image (step size = 7)
        # x = (4, 3600), dict_keys(['rangex', 'rangey', 'offset', 'sx', 'sy', 'stepsize', 'winsize', 'follow_matlab'])
        X, L =pa2.getfeatures(img, 7)

        if normalization:
            X = zero_one_normalization(X)

        if algorithm == 'kmeans':
            initial_centers = np.ndarray((clusters, X.shape[0]))
            for i in range(X.shape[0]):
                initial_centers[:, i] = np.random.uniform(np.min(X[i, :]), np.max(X[i, :]), clusters).reshape((1, clusters))

            _, Z = cluster.kmeans( vq.whiten(X.T), initial_centers, iterations)

        if algorithm == 'EM-GMM':
            initial_pi = np.array([1/clusters for _ in range(clusters)])
            print(initial_pi)
            initial_mean = np.random.rand(clusters, X.shape[0])
            initial_covariance =np.array([np.eye(X.shape[0])] * clusters)
            
            pi, mean, _, Z = cluster.EM_GMM(vq.whiten(X.T), initial_pi, initial_mean,initial_covariance, iterations)
        
        if algorithm == 'Mean-shift':
            peaks, Z = cluster.mean_shift(vq.whiten(X.T), bandwidth, 0.01, 0.5)

        Y= []
        for i in range(X.shape[1]):
            if algorithm == 'Mean-shift':
                Y.append(int(Z[i]))
            else:
                Y.append(np.argmax(Z[i, :]) + 1)
        Y = np.array(Y)
        print(Y)

        # make segmentation image from labels
        segm = pa2.labels2seg(Y, L)
        pl.subplot(len(images),3,3*images.index(image)+2)
        # pl.axis('off')
        pl.imshow(segm)

        # color the segmentation image
        csegm = pa2.colorsegms(segm, img)
        pl.subplot(len(images),3,3*images.index(image)+3)
        # pl.axis('off')
        pl.imshow(csegm)
    
    pl.subplots_adjust(left=0.10, top= 0.90, right = 0.90, bottom = 0.10, wspace = 0.05, hspace = 0.4)
    if algorithm != 'Mean-shift':
        pl.suptitle(f'Algorithm: {algorithm}, iteration: {iterations}, clusters(K): {clusters}, normalization:{normalization}')
    else:
        pl.suptitle(f'Algorithm: {algorithm}, bandwidth: {bandwidth}, normalization: {normalization}')
    pl.show()

def zero_one_normalization(data):
    """Do zero-one normalization 

    Parameters
    ----------
    data
        input data

    Returns
    -------
        normalized input data
    """    
    print('********** start data normalization **********')

    x0_max, x0_min = data[0, :].max(), data[0, :].min()
    print(x0_max, x0_min)
    x1_max, x1_min = data[1, :].max(), data[1, :].min()
    print(x1_max, x1_min)
    x2_max, x2_min = data[2, :].max(), data[2, :].min()
    print(x2_max, x2_min)
    x3_max, x3_min = data[3, :].max(), data[3, :].min()
    print(x3_max, x3_min)
    for i in range(data.shape[1]):
        data[0, i] = (data[0, i] - x0_min) /x0_max - x0_min
        data[1, i] = (data[1, i] - x1_min) /x1_max - x1_min
        data[2, i] = (data[2, i] - x2_min) /x2_max - x2_min
        data[3, i] = (data[3, i] - x3_min) /x3_max - x3_min

    print('********** end data normalization **********')
    return data


if __name__ == '__main__':
    plot('kmeans', 100, clusters = 5, normalization = True)
    # plot('EM-GMM', 300, clusters = 8, normalization = False)
    # plot('Mean-shift', bandwidth = 0.5, normalization = True)

