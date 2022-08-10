'''
@Auther: Shixiang WANG
@EID: sxwang6
@Date: 08/03/2022

@Discription:
This class is responsible for loading data
'''

import numpy as np

def data_loader(data_name)  -> tuple[np.ndarray, np.ndarray]:
    """Load data

    Parameters
    ----------
    data_name
       String: Only three choices: 'A', 'B', and'C'

    Returns
    -------
        ndarray(dimensions, points index): data points, ndarry(dimensiosn, points index): labels
    """    

    file_path_X = 'data/cluster_data_text/cluster_data_data' + data_name + '_X.txt'
    file_path_Y = 'data/cluster_data_text/cluster_data_data' + data_name + '_Y.txt'
    
    points, labels = [], []

    fx = open(file_path_X)
    for line in fx:
        tmp = []
        for val in line.split():
            tmp.append(float(val))
        points.append(tmp)
    fx.close()

    fy = open(file_path_Y)
    for line in fy:
        tmp = []
        for label in line.split():
            tmp.append(float(label))
        labels.append(tmp)
    fy.close

    return np.transpose(np.array(points)), np.transpose(np.array(labels))


