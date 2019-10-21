import numpy as np
from scipy.optimize import minimize

import torch


def direction(theta, phi):
    '''Return the direction vector of a cylinder defined
    by the spherical coordinates theta and phi.
    '''
    return np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta),
                     np.cos(theta)])

def projection_matrix(w):
    I_m = torch.eye(3)
    device = w.device
    I_m = I_m.to(device)
    mult = torch.mm(w, w.t())
    return I_m - mult

def projection_matrix_np(w):
    '''Return the projection matrix  of a direction w.'''
    return np.identity(3) - np.dot(np.reshape(w, (3,1)), np.reshape(w, (1, 3)))

def skew_matrix(w):
    '''Return the skew matrix of a direction w.'''
    return torch.tensor([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], device=w.device)

def skew_matrix_np(w):
    '''Return the skew matrix of a direction w.'''
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def calc_A_np(Ys):
    '''Return the matrix A from a list of Y vectors.'''
    return sum(np.dot(np.reshape(Y, (3,1)), np.reshape(Y, (1, 3)))
            for Y in Ys)

def calc_A(Ys):
    '''Return the matrix A from a list of Y vectors.'''
    return sum( [ torch.mm(Y, Y.t())
            for Y in Ys.split(1, dim=1)])

def calc_A_hat(A, S):
    '''Return the A_hat matrix of A given the skew matrix S'''
    return torch.mm(S, torch.mm(A, S.t()))

def calc_A_hat_np(A, S):
    '''Return the A_hat matrix of A given the skew matrix S'''
    return np.dot(S, np.dot(A, np.transpose(S)))

def preprocess_data_np(Xs_raw):
    '''Translate the center of mass (COM) of the data to the origin.
    Return the prossed data and the shift of the COM'''
    n = len(Xs_raw)
    Xs_raw_mean = sum(X for X in Xs_raw) / n

    return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean

def preprocess_data(Xs_raw):
    '''Translate the center of mass (COM) of the data to the origin.
    Return the prossed data and the shift of the COM'''
    X_mean = torch.mean(Xs_raw,0)
    Xs_raw= Xs_raw - X_mean.expand_as(Xs_raw)
    return Xs_raw, X_mean

def G(w, Xs):
    '''Calculate the G function given a cylinder direction w and a
    list of data points Xs to be fitted.'''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = torch.mm(P, Xs.t())
    Ys = Ys.to(Xs.device)
    A = calc_A(Ys)
    A = A.to(Xs.device)
    A_hat = calc_A_hat(A, skew_matrix(w))
    A_hat = A_hat.to(Xs.device)
    u = sum(torch.mm(Y.t(), Y) for Y in Ys.split(1, dim=1)) / n
    v = torch.mm(A_hat, sum(torch.mm(Y.t(), Y) * Y for Y in Ys.split(1, dim=1))) / torch.trace(torch.mm(A_hat, A))

    return sum((torch.mm(Y.t(), Y) - u - 2 * torch.mm(Y.t(), v)) ** 2 for Y in Ys.split(1, dim=1))

def G_np(w, Xs):
    '''Calculate the G function given a cylinder direction w and a
    list of data points Xs to be fitted.'''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = [np.dot(P, X) for X in Xs]
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))

    
    u = sum(np.dot(Y, Y) for Y in Ys) / n
    v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / (np.trace(np.dot(A_hat, A)))

    return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)

def C(w, Xs):
    '''Calculate the cylinder center given the cylinder direction and 
    a list of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = torch.mm(P, Xs.t())
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))
    return torch.mm(A_hat, sum(torch.mm(Y.t(), Y) * Y for Y in Ys.split(1, dim=1))) / torch.trace(torch.mm(A_hat, A))

def C_np(w, Xs):
    '''Calculate the cylinder center given the cylinder direction and 
    a list of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = [np.dot(P, X) for X in Xs]
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))

    return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(np.dot(A_hat, A))

def r(w, Xs, c):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    Xs_off = c - Xs.t()
    Xs_off = Xs_off.split(1, dim=1)
    return torch.sqrt(sum(torch.mm(X.t(),torch.mm(P, X)) for X in Xs_off) / n)

def r2(w, Xs, c):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    Xs_off = c - Xs.t()
    Xs_off = Xs_off.split(1, dim=1)

    vector_r = [torch.mm(X.t(),torch.mm(P, X)) for X in Xs_off]
    r = torch.sqrt(sum(vector_r) / n)

    return torch.tensor(vector_r), r

def r_np(w, Xs):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    c = C(w, Xs)

    return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

def H(c, Xs):
    '''Calculate the height given the cylinder center and a list
    of data points.
    '''
    
    distances = torch.sqrt(torch.sum((Xs.t() - c)**2, 0))
    return 2 * torch.mean(distances)

def H_np(c, Xs):
    '''Calculate the height given the cylinder center and a list
    of data points.
    '''
    distances = [np.sqrt(np.dot(X - c, X - c)) for X in Xs]
    return 2 * np.mean(distances)

def fit_tw(data, guess_angle):
    '''Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from 
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf

    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction
    
    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    '''
    Xs, t = preprocess_data(data)  

    # Set the start points
    w = guess_angle
    center = C(w, Xs)
    print(center.shape)
    height = H(center, Xs)
    R = r(w, Xs, center)
    error = G(w, Xs)
    return w, center.t() + t, R, height, error

def fit_t(data, guess_angles=None):
    '''Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from 
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf

    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction
    
    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    '''
    Xs, t = preprocess_data(data)  

    # Set the start points
    w = PCA(Xs, 1)
    center = C(w,   Xs)
    height = H(center, Xs)
    R = r(w, Xs, center)
    error = G(w, Xs)
    return w, center.t() + t, R, height, error

def fit_np(data, guess_angles=None):
    '''Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from 
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf

    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction
    
    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    '''
    Xs, t = preprocess_data(data)  

    # Set the start points

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    # Fit the cylinder from different start points 

    best_fit = None
    best_score = float('inf')

    for sp in start_points:
        fitted = minimize(lambda x : G(direction(x[0], x[1]), Xs),
                    sp, method='Powell', tol=1e-6)

        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    w = direction(best_fit.x[0], best_fit.x[1])

    center = C(w, Xs)
    height = H(center, Xs)
    R = r(w, Xs)
    
    return w, center + t, R, height, best_fit.fun 

def PCA(X, k=2):
     # preprocess the data
     
     # svd
     U,S,V = torch.svd(torch.t(X))
     return U[:,:k]


def r_individual(data):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    Xs, t = preprocess_data(data)
    w = PCA(Xs, 1)
    c = C(w,  Xs)
    P = projection_matrix(w)
    Xs_off = c - Xs.t()

    Xs_off = Xs_off.split(1, dim=1)
    vector = torch.sqrt(torch.tensor([torch.mm(X.t(), torch.mm(P, X)) for X in Xs_off], device=data.device))
    # print(vector)

    return vector
