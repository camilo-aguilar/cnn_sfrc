
# Python program to illustrate the concept 
# of threading 
# importing the threading module 
from fitting import fit
from fit_torch import fit_t
import torch
import numpy as np
import math
import time

from skimage import measure
from skimage.morphology import watershed, binary_erosion, ball, binary_dilation, binary_opening 
from scipy import ndimage as ndi

import multiprocessing
from functools import partial
from contextlib import contextmanager

def guess_cylinder_parameters_w_torch(L, pre_seg):
    indexes = (pre_seg == L).nonzero().float()
    fiber_points = indexes
    w_fit, C_fit, r_fit, h_fit, fit_err  = fit_t(fiber_points)


    w_fit = w_fit[:, 0].cpu().numpy()
    C_fit = C_fit[0].cpu().numpy()
    r_fit = r_fit.item()
    h_fit = h_fit.item()
    fit_err = fit_err.item()

    Txy = np.arctan2(w_fit[1], w_fit[0]) * 180 / np.pi
    if(Txy < 0):
        Txy = 180 + Txy
    Tz = np.arccos(np.dot(w_fit, np.array([0, 0, 1])) / np.linalg.norm(w_fit, 2)) * 180 / np.pi


    return [L.item(), C_fit[0], C_fit[1], C_fit[2], r_fit, h_fit, Txy, Tz, fit_err]


def fit_all_torch(volume, offset_coordinates=None):
    labels = torch.unique(volume)
    results = []
    for i in (labels):
        if(i == 0 or i==1):
            continue
        results.append(guess_cylinder_parameters_w_torch(i, volume[0, 0, ...]))
    return results


def guess_cylinder_parameters_indexes(indexes):
    step_size = 1
    num_points = len(indexes[0])
    if(num_points < 20):
        x = indexes[0].mean()
        y = indexes[1].mean()
        z = indexes[2].mean()
        return (-1, [x, y, z], -1, -1, -1, -1, -1)
    if(num_points > 20):
        step_size = int(math.floor(num_points / 20))

    fiber_points = [np.array([indexes[0][k], indexes[1][k], indexes[2][k]]).astype(np.float) for k in range(1, num_points, step_size)]

    w_fit, C_fit, r_fit, h_fit, fit_err = fit(fiber_points)
    Txy = np.arctan2(w_fit[1], w_fit[0]) * 180 / np.pi
    if(Txy < 0):
        Txy = 180 + Txy
    Tz = np.arccos(np.dot(w_fit, np.array([0, 0, 1])) / np.linalg.norm(w_fit, 2)) * 180 / np.pi

    return(-1, C_fit, r_fit, h_fit, Txy, Tz, fit_err)

def guess_cylinder_parameters_w(L, pre_seg):
    L = L.item()
    indexes = np.where(pre_seg == L)

    num_points = len(indexes[0])
    if(num_points > 30 ):
        step_size = int(math.floor(num_points / 30))
    else:
        return  (-1, np.array([0 ,0, 0]))

    idx = np.random.choice(num_points, 30, replace=False)
    fiber_points = [np.array([indexes[0][k], indexes[1][k], indexes[2][k]]).astype(np.float) for k in idx]


    w_fit, C_fit, r_fit, h_fit, fit_err = fit(fiber_points)

    Xs_raw = fiber_points
    n = len(Xs_raw)
    Xs_raw_mean = sum(X for X in Xs_raw) / n

    fiber_points = [X - Xs_raw_mean for X in Xs_raw]

    return(L, w_fit)

def guess_cylinder_parameters_voids(L, pre_seg):
    # L = L
    indexes = np.where(pre_seg == L)

    step_size = 1
    num_points = len(indexes[0])
    if(num_points < 5):
        x = indexes[0].mean()
        y = indexes[1].mean()
        z = indexes[2].mean()
        return [L, x, y, z, -1, -1, -1, -1, -1]
    if(num_points > 5):
        step_size = int(math.floor(num_points / 5))

    fiber_points = [np.array([indexes[0][k], indexes[1][k], indexes[2][k]]).astype(np.float) for k in range(1, num_points, step_size)]

    w_fit, C_fit, r_fit, h_fit, fit_err = fit(fiber_points)

    return [L, C_fit[0], C_fit[1], C_fit[2], r_fit, num_points, w_fit[0], w_fit[1], w_fit[2]]


def fit_all_voids_parallel(volume, offset_coordinates=None):
    labels = np.unique(volume)
    names = labels[2:]

    number = multiprocessing.cpu_count()
    with poolcontext(processes=number) as pool:
        results = pool.map(partial(guess_cylinder_parameters_voids, pre_seg=volume), names)
      
    return results

def guess_cylinder_parameters_simple(L, indxs):
    L = L.cpu().item()
    fiber_points = indxs[L]
    w_fit, C_fit, r_fit, h_fit, fit_err = fit(fiber_points)
    Txy = np.arctan2(w_fit[1], w_fit[0]) * 180 / np.pi
    if(Txy < 0):
        Txy = 180 + Txy
    Tz = np.arccos(np.dot(w_fit, np.array([0, 0, 1])) / np.linalg.norm(w_fit, 2)) * 180 / np.pi

    return [L, C_fit[0], C_fit[1], C_fit[2], r_fit, h_fit, Txy, Tz, fit_err]


def guess_cylinder_parameters(L, pre_seg):
    L = L.item()
    indexes = np.where(pre_seg == L)

    step_size = 1
    num_points = len(indexes[0])
    if(num_points < 30):
        x = indexes[0].mean()
        y = indexes[1].mean()
        z = indexes[2].mean()
        return [L, x, y, z, -1, -1, -1, -1, -1]
    if(num_points > 30):
        step_size = int(math.floor(num_points / 30))

    fiber_points = [np.array([indexes[0][k], indexes[1][k], indexes[2][k]]).astype(np.float) for k in range(1, num_points, step_size)]

    w_fit, C_fit, r_fit, h_fit, fit_err = fit(fiber_points)
    Txy = np.arctan2(w_fit[1], w_fit[0]) * 180 / np.pi
    if(Txy < 0):
        Txy = 180 + Txy
    Tz = np.arccos(np.dot(w_fit, np.array([0, 0, 1])) / np.linalg.norm(w_fit, 2)) * 180 / np.pi

    return [L, C_fit[0], C_fit[1], C_fit[2], r_fit, h_fit, Txy, Tz, fit_err]


def guess_cylinder_parameters_merged(L, L2, pre_seg):
    indexes_a = np.where(pre_seg == L)
    if(L2 > 0):
        indexes_b = np.where(pre_seg == L2)
        indexes = (np.concatenate((indexes_a[0], indexes_b[0])), np.concatenate((indexes_a[1], indexes_b[1])), np.concatenate((indexes_a[2], indexes_b[2]))) 
    else:
        indexes = indexes_a

    step_size = 1
    num_points = len(indexes[0])
    if(num_points < 30):
        x = indexes[0].mean()
        y = indexes[1].mean()
        z = indexes[2].mean()
        return [L, x, y, z, -1, -1, -1, -1, -1]
    if(num_points > 30):
        step_size = int(math.floor(num_points / 30))

    fiber_points = [np.array([indexes[0][k], indexes[1][k], indexes[2][k]]).astype(np.float) for k in range(1, num_points, step_size)]

    w_fit, C_fit, r_fit, h_fit, fit_err = fit(fiber_points)
    Txy = np.arctan2(w_fit[1], w_fit[0]) * 180 / np.pi
    if(Txy < 0):
        Txy = 180 + Txy
    Tz = np.arccos(np.dot(w_fit, np.array([0, 0, 1])) / np.linalg.norm(w_fit, 2)) * 180 / np.pi

    return [L, C_fit[0], C_fit[1], C_fit[2], r_fit, h_fit, Txy, Tz, fit_err]


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def fit_all_fibers_parallel_simple(volume, offset_coordinates=None):
    labels = np.unique(volume)

    names = labels[1:]

    number = multiprocessing.cpu_count()

    with poolcontext(processes=number) as pool:
        results = pool.map(partial(guess_cylinder_parameters_w, pre_seg=volume), names)
    return results

def fit_all_fibers_parallel_from_torch(volume, offset):
    labels = torch.unique(volume)
    list_of_indxs = {}
    for f_id in labels:
        if(f_id == 0 or f_id == 1):
            continue
        coordinates = (volume == f_id).nonzero().cpu().split(1, dim=1)
        step_size = 1
        num_points = len(coordinates[0])
        if(num_points < 30):
            volume[coordinates] = 0
            continue

        if(num_points > 30):
            step_size = int(math.floor(num_points / 30))

        # list_of_indxs[f_id] = [np.array([coordinates[0][k].float() + offset[0], coordinates[1][k].float() + offset[1], coordinates[2][k].float() + offset[2]]).astype(np.float) for k in range(1, num_points, step_size)]
        list_of_indxs[f_id] = coordinates
        print(f_id)
    number = multiprocessing.cpu_count()
    with poolcontext(processes=number) as pool:
        results = pool.map(partial(guess_cylinder_parameters_simple, indxs=list_of_indxs), labels)

    return results

'''
    volume shape: [chs rws cls slcs]
'''
def fit_all_fibers_parallel(volume, offset_coordinates=None):
    labels = np.unique(volume)
    names = labels[2:]

    number = multiprocessing.cpu_count()
    # print(number)
    # print(names.dtype)
    with poolcontext(processes=number) as pool:
        results = pool.map(partial(guess_cylinder_parameters, pre_seg=volume), names)
      
    return results

'''
    result_dict = {}
    list_of_ids_unassigned = set()

    for el in results:
        if(el[0] == 1):
            idx = np.where(volume == el[0])
            volume[idx] = 0
        elif(el[6] == -1):
            idx = np.where(volume == el[0])
            volume[idx] = 0

        elif(el[6] > 100 and el[0] > 1):
            idx = np.where(volume == el[0])
            volume_to_correct[idx] = el[0]
            list_of_ids_unassigned.add(el[0])
        
        else:
            result_dict[el[0]] = (el[1] + offset_coordinates, el[2], el[3], el[4], el[5], el[6])
    
    (v2, fibers_corrected) = post_processing(volume_to_correct)
    for i in fibers corrected:
        if(i == 0):
            continue
        idx = np.where(volume_to_correct == i)
        if(len(list_of_ids_unassigned)):
            result_dict[list_of_ids_unassigned.pop()] =  
  '''
def post_processing_simple(list_of_fibers, labels):
    final_list = []
    volume_to_correct = np.zeros(labels.shape)
    counter = 1
    from_value = [0]
    to_value = [0]
    for el in list_of_fibers:
        if(el[0] == 1):
            idx = np.where(labels == el[0])
            labels[idx] = 0
        elif(el[6] == -1):
            idx = np.where(labels == el[0])
            labels[idx] = 0

        elif(el[6] > 100 and el[0] > 1):
            idx = np.where(labels == el[0])
            volume_to_correct[idx] = el[0]
            labels[idx] = 0
        else:
            final_list.append([counter, el[1][0], el[1][1], el[1][2], el[2], el[3], el[4], el[5], el[6]])
            from_value.append(el[0])
            to_value.append(counter)
            counter = counter + 1
    for i in range(len(from_value)):
        labels[np.where(labels == from_value[i])] = to_value[i]
        
    return final_list


def post_processing_list(list_of_fibers, labels):
    final_list = []
    volume_to_correct = np.zeros(labels.shape)
    counter = 2
    from_value = [1]
    to_value = [1]
    for el in list_of_fibers:
        if(el[0] == 1):
            idx = np.where(labels == el[0])
            labels[idx] = 1
        elif(el[6] == -1):
            idx = np.where(labels == el[0])
            labels[idx] = 1

        elif(el[6] > 100 and el[0] > 1):
            idx = np.where(labels == el[0])
            volume_to_correct[idx] = el[0]
            labels[idx] = 1
        else:
            final_list.append([counter, el[1][0], el[1][1], el[1][2], el[2], el[3], el[4], el[5], el[6]])
            from_value.append(el[0])
            to_value.append(counter)
            counter = counter + 1

    se = ball(1)
    im = binary_erosion(volume_to_correct > 0, se)
    im = binary_erosion(im > 0, se)
    temp_labels, temp_nums = measure.label(im, return_num=True)
    if(temp_nums > 1):
        histogram = np.histogram(temp_labels, temp_nums)
        to_clean = histogram[1][np.where(histogram[0] < 10)]
        
        for i in to_clean:
            temp_labels[np.where(temp_labels == i)] = 0
        distance = ndi.distance_transform_edt(labels)
        ws = watershed(-distance, temp_labels, mask=(volume_to_correct > 0))
        
        # init_counter = counter
        corrected_list = fit_all_fibers_parallel(ws)
        for el in corrected_list:
            if(el[6] > -1 and el[6] < 100):
                idx = np.where(ws == el[0])
                new_id = from_value[-1] + 1
                labels[idx] = new_id 
                volume_to_correct[idx] = 0
                final_list.append([counter, el[1][0], el[1][1], el[1][2], el[2], el[3], el[4], el[5], el[6]])
                from_value.append(new_id)
                to_value.append(counter)
                counter = counter + 1

    # distance = ndi.distance_transform_edt(labels > 0)
    # ws = watershed(-distance, labels, mask=(labels > 0))

    to_value = np.array(to_value)
    from_value = np.array(from_value)

    for i in range(len(from_value)):
        labels[np.where(labels == from_value[i])] = to_value[i]
    return final_list, volume_to_correct


def post_processing(list_of_fibers, labels):
    print("Post Processing")
    final_dict = {}
    volume_to_correct = np.zeros(labels.shape)
    counter = 2

    fibers_corrected = 0
    fibers_small = 0
    for el in list_of_fibers:
        if(el[0] == 1):
            idx = np.where(labels == el[0])
            labels[idx] = 1
        elif(el[8] == -1):
            print("Fiber {} has a problem".format(el[0]))
            idx = np.where(labels == el[0])
            labels[idx] = 1
            fibers_corrected += 1

        elif(el[8] > 100 and el[0] > 1):
            print("Fiber {} has a problem".format(el[0]))
            idx = np.where(labels == el[0])
            volume_to_correct[idx] = el[0]
            labels[idx] = 1
            fibers_corrected +=1 
        else:
            final_dict[el[0]] = el

    print("Coorrected")
    print(fibers_corrected)

    print("Fibers Small")
    print(fibers_small)
    return final_dict, volume_to_correct

def PCA(X, k=2):
     # preprocess the data
     X_mean = torch.mean(X,0)
     X = X - X_mean.expand_as(X)
    
     # svd
     U,S,V = torch.svd(torch.t(X))
     return U[:,:k]




def H(c, Xs):
    '''Calculate the height given the cylinder center and a list
    of data points.
    '''
    distances = [np.sqrt(np.dot(X - c, X - c)) for X in Xs]
    return 2 * np.mean(distances)


import torch.nn as nn


class autoencoder(nn.Module):
    def __init__(self, iput_dims, output_dims):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
             nn.Linear(iput_dims * 3, 256),
             nn.Linear(256, 128),
             nn.Linear(128, output_dims),
             )

    def forward(self, x):
        x = self.encoder(x)
        return x


import torch
def train_autoencoder():

    num_epochs = 1000
    net = autoencoder(30, 3).cuda()
    net.load_state_dict(torch.load('./sim_autoencoder.pth'))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
    net.parameters(), lr=0.001, weight_decay=1e-5)


    masks = tensors_io.load_volume_uint16('../../updated_fibers/UPDATED_TRAINING_LABELS', scale=1).long().unsqueeze(0)
    for epoch in range(num_epochs):
        mini_M, _ = tensors_io.random_crop_3D_image_batched(masks, masks, 64)
        list_l = fit_all_fibers_parallel_simple(mini_M[0,0,...].numpy())

        x = []
        y = []
        for el in list_l:
            if el[0].sum() == 0:
                continue
            else:
                y.append(el[0])
                x.append(el[1])
        y = torch.tensor(y).float().cuda()
        x = torch.tensor(x).float().cuda()
        x = x.view(x.shape[0], -1)
        # ===================forward=====================
        x_hat = net(x)
        loss = criterion(x_hat, y)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))

    torch.save(net.state_dict(), './sim_autoencoder.pth')


if __name__ == '__main__':
    import tensors_io
    train_autoencoder()
    # Python program to understand  
    # the concept of pool 


    '''
    masks = tensors_io.load_volume_uint16('../../updated_fibers/UPDATED_TRAINING_LABELS', scale=1).long().unsqueeze(0)
    


    mini_M, _ = tensors_io.random_crop_3D_image_batched(masks, masks, 64)
    mini_M1 = mini_M[0,0,...].numpy()

    start = time.time()

    start = time.time()
    list_l = fit_all_fibers_parallel_simple(mini_M1)
    print(len(list_l[0][1]))
    x = []
    y = []
    for el in list_l:
        if el[0].sum() == 0:
            continue
        else:
            y.append(el[0])
            x.append(el[1])
    y = torch.tensor(y)
    x = torch.tensor(x)


    print(x.mean(1))
    x = x.view(x.shape[0], -1)
    print(y.shape)
    print(x.shape)
    nums = len(list_l)
    '''
    '''
    mini_M = masks # tensors_io.random_crop_3D_image_batched(masks, masks, 64)
    [bz, ch, rows, cols, slices] = masks.shape
    mini_M = mini_M[0,0,...].numpy()
    
    output_image = np.zeros([3, rows, cols, slices])
    for i in np.unique(masks):
        if(i == 0):
            continue
        print("Fitting Fiber {}".format(i))
        (pp, indices) = guess_cylinder_parameters_w(i, mini_M)
        w = pp
        w = ((w + 1) / 2) 
        output_image[0, indices[0][:],indices[1][:], indices[2][:]] = w[0]
        output_image[1, indices[0][:],indices[1][:], indices[2][:]] = w[1]
        output_image[2, indices[0][:],indices[1][:], indices[2][:]] = w[2]
    tensors_io.save_subvolume(torch.from_numpy(output_image).float(), 'direction_training')
        
    '''




    # 0 -> - 1
    # 125 -> 0
    # 255 -> 1



    '''
    tensors_io.save_subvolume_color(mini_M.float() * 0, (mini_M == mini_M.max()), 'test', num_classes=2)
    list_l = fit_all_fibers_parallel(mini_M)
    
    # print(list_r)
    with open('list.txt', 'w') as output:
        for element in list_l:
            to_write = "{},{},{},{},{},{}, {}, {}\n".format(element[0], element[1][0], element[1][1], element[1][2], element[2], element[3] * 2, element[4] * 180 / 3.14159, element[5] * 180 / 3.14159)
            output.write(to_write)
            output.write("\n")

    
    print(list_l[-1])
    '''
    '''
    print("Starting Second Step")
    start2 = time.time()
    list_of_cylinders = []
    for i in labels:
        list_of_parameters = guess_cylinder_parameters(mini_M, i.item())
        list_of_cylinders.append(list_of_parameters)
    end2 = time.time()

    print(results)
    print(list_of_cylinders)
    print("Time 1")
    print(end - start, 's')
    print("Time 2")
    print(end2 - start2, 's')
'''
'''
        # input list 
        mylist = [1,2,3,4,5] 
      
        # creating a pool object 
        p = multiprocessing.Pool() 
      
        # map list to target function 
        result = p.map(square, mylist) 
      
        print(result) 


   


    import multiprocessing
    from functools import partial
    from contextlib import contextmanager

    mini_M = mini_M[0, ...].numpy()
    list_of_cylinders = []
    for i in labels:
        if(i == 0):
            continue

        list_of_parameters = guess_cylinder_parameters(mini_M, i.item())
        list_of_cylinders.append(list_of_parameters)



# importing the multiprocessing module 
import multiprocessing   
    # creating processes 
p1 = multiprocessing.Process(target=guess_cylinder_parameters, args=(mini_M, 10))
p2 = multiprocessing.Process(target=guess_cylinder_parameters, args=(mini_M, 20))

# starting process 1 
p1.start() 
# starting process 2 
p2.start() 

# wait until process 1 is finished 
p1.join() 
# wait until process 2 is finished 
p2.join() 
`
# both processes finished 
print("Done!") 

end = time.time()
print("Time")
print(end - start, 's')

'''

