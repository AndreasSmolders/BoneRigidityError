import torch
import numpy as np
def BRE_torch(X, Y):
    assert X.shape == Y.shape
    assert X.shape[0] == Y.shape[0] == 3
    centroid_X = torch.mean(X, axis=1)
    centroid_Y = torch.mean(Y, axis=1)
    Xm = X - centroid_X
    Ym = Y - centroid_Y

    H = Xm @ torch.transpose(Ym,0,1)
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    if torch.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_X + centroid_Y
    BRE = torch.mean(torch.linalg.norm(Y-(R@X+t),axis=0)))  
    return BRE

def BRE_numpy(X, Y):
    assert X.shape == Y.shape
    assert X.shape[0] == Y.shape[0] == 3
    centroid_X = np.mean(X, axis=1)
    centroid_Y = np.mean(Y, axis=1)
    Xm = X - centroid_X
    Ym = Y - centroid_Y

    H = Xm @ np.transpose(Ym,0,1)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_X + centroid_Y
    BRE = np.mean(np.linalg.norm(Y-(R@X+t),axis=0)))  
    return BRE
