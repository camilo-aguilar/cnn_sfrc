import numpy as np
#import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage.morphology import dilation, ball

from .geometry import rotation_matrix_from_axis_and_angle 
from . import fitting


def show_G_distribution(data):
    '''Show the distribution of the G function.'''
    Xs, t = fitting.preprocess_data(data)  

    Theta, Phi = np.meshgrid(np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 50))
    G = []

    for i in range(len(Theta)):
        G.append([])
        for j in range(len(Theta[i])):
            w = fitting.direction(Theta[i][j], Phi[i][j])
            G[-1].append(fitting.G(w, Xs))

    plt.imshow(G, extent=[0, np.pi, 0, 2 * np.pi], origin='lower')
    plt.show()

def show_fit(w_fit, C_fit, r_fit, Xs):
    '''Plot the fitting given the fitted axis direction, the fitted
    center, the fitted radius and the data points.
    '''

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot the data points
    
    ax.scatter([X[0] for X in Xs], [X[1] for X in Xs], [X[2] for X in Xs])
   
    # Get the transformation matrix

    theta = np.arccos(np.dot(w_fit, np.array([0, 0, 1])))
    phi = np.arctan2(w_fit[1], w_fit[0])

    M = np.dot(rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
               rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), theta))

    # Plot the cylinder surface
   
    delta = np.linspace(-np.pi, np.pi, 20)
    z = np.linspace(-50,50, 20)

    Delta, Z = np.meshgrid(delta, z)
    X = r_fit * np.cos(Delta)
    Y = r_fit * np.sin(Delta)

    for i in range(len(X)):
        for j in range(len(X[i])):
            p = np.dot(M, np.array([X[i][j], Y[i][j], Z[i][j]])) + C_fit

            X[i][j] = p[0]
            Y[i][j] = p[1]
            Z[i][j] = p[2]

    ax.plot_surface(X, Y, Z, alpha=0.2)

    # Plot the center and direction

    ax.quiver(C_fit[0], C_fit[1], C_fit[2], 
            r_fit * w_fit[0], r_fit * w_fit[1], r_fit * w_fit[2], color='red')


    plt.show()
    
    
def draw_fit(img, w_fit, C_fit, r_fit, l_fit):
    '''Plot the fitting given the fitted axis direction, the fitted
    center, the fitted radius and the data points.
    '''

   
    # Get the transformation matrix

    theta = np.arccos(np.dot(w_fit, np.array([0, 0, 1])))
    phi = np.arctan2(w_fit[1], w_fit[0])

    M = np.dot(rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
               rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), theta))

    # Plot the cylinder surface
   
    delta = np.linspace(-np.pi, np.pi, 10)
    z = np.linspace(-l_fit,l_fit, 100)

    Delta, Z = np.meshgrid(delta, z)
    X = 1 * np.cos(Delta)
    Y = 1 * np.sin(Delta)

    for i in range(len(X)):
        for j in range(len(X[i])):
            p = np.floor(np.dot(M, np.array([X[i][j], Y[i][j], Z[i][j]])) + C_fit).astype(np.int)

            flag_bounds = 1
            
            # Check for out of bounds stuff
            for k in range(len(img.shape)):
                if(p[k] < 0 or p[k] >= img.shape[k]):
                    flag_bounds = 0

            # Fill out the image
            if(flag_bounds):
                img[p[0],p[1],p[2]] = 1
            
    selem = ball(3)
    img = dilation(img, selem)


    return img

def cylinder_level_set(image_shape, center=None, radius=None):
    """Create a circle level set with binary values.

    Parameters
    ----------
    image_shape : tuple of positive integers
        Shape of the image
    center : tuple of positive integers, optional
        Coordinates of the center of the circle given in (row, column). If not
        given, it defaults to the center of the image.
    radius : float, optional
        Radius of the circle. If not given, it is set to the 75% of the
        smallest image dimension.

    Returns
    -------
    out : array with shape `image_shape`
        Binary level set of the circle with the given `radius` and `center`.

    See also
    --------
    checkerboard_level_set
    """

    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid)**2, 0))
    res = np.int8(phi > 0)
    return res


def show_fit2(phi1, phi2):
    '''Plot the fitting given the fitted axis direction, the fitted
    center, the fitted radius and the data points.
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    (X1,Y1,Z1) = np.where(phi1 > 0)
    (X2,Y2,Z2) = np.where(phi2 > 0)

    ax.scatter(X1, Y1, Z1, alpha=0.2, color='red')
    ax.scatter(X2, Y2, Z2, alpha=0.2, color='blue')

    # Plot the center and direction
    plt.show()