from __future__ import print_function
import tensors_io
from skimage import measure
from unet.cylinder_fitting import fit_all_fibers_parallel, fit_all_voids_parallel, fit_all_fibers_parallel_from_torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import unet.cylinder_fitting.fit_torch as fit_t
from collections import defaultdict
import torch
import math
import time
from skimage.morphology import watershed, binary_erosion, ball, binary_dilation, binary_opening



################################  Manipulate Volumes ################################
def fit_long_fibers(h5_volume_dir, dataset_name):
    print("Fitting Very Long Fibers")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    complete_fiber_set = set()
    edge_fiber_set = set()
    with open("fiber_dictionary_CORRECTED_EDGE.txt") as f:
        for line in f.readlines():
            edge_fiber_set.add(int(line))
    print(edge_fiber_set)
    fiber_dictionary = dict()
    Volume = torch.from_numpy(Volume)
    list_fibers = get_fiber_properties_post_processing(Volume, offset=[0, 0, 0], complete_fibers=complete_fiber_set, edge_fiber_set={}, fibers_to_label=edge_fiber_set)

    for f_id in list_fibers.keys():
        fiber_dictionary[f_id] = list_fibers[f_id]

    f = open(h5_volume_dir + "/fiber_dictionary_CORRECTED_EDGES.txt", "w")
    for k in fiber_dictionary.keys():
        el = fiber_dictionary[k]
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.0f},{:.0f},{:.2f}\n".format(el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7], el[8]))
    f.close()


def get_whole_volume_statistics(h5_volume_dir, dataset_name, window_size=500, device=None):
    print("Getting Fibers Statistics")
    if(device is None):
        device = torch.device("cuda:0")
    fiber_dictionary = {}
    edge_fiber_set = set()
    complete_fiber_set = set()
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    rows, cols, slices = Volume.shape
    print(Volume.shape)
    start_x = []
    start_y = []
    start_z = []

    sz = 0
    while(sz + window_size < rows):
        start_x.append(sz)
        sz = sz + window_size / 2

    sz = 0
    while(sz + window_size < cols):
        start_y.append(sz)
        sz = sz + window_size / 2

    sz = 0
    while(sz + window_size < slices):
        start_z.append(sz)
        sz = sz + window_size / 2

    num_partitions = len(start_z) * len(start_y) * len(start_x)
    counter = 0
    print("Starting Fitting")

    for x in start_x:
        for y in start_y:
            for z in start_z:
                print("Partition {} done out of {}".format(counter, num_partitions))
                temp_vol = Volume[x:x + window_size, y:y + window_size, z:z + window_size]
                temp_vol = torch.from_numpy(temp_vol).to(device)
                list_fibers = get_fiber_properties_post_processing(temp_vol, [x, y, z], complete_fibers=complete_fiber_set, edge_fiber_set=edge_fiber_set, vol_dim=Volume.shape)

                for f_id in list_fibers.keys():
                    fiber_is_complete = 1 - list_fibers[f_id][-1]

                    if(fiber_is_complete):
                        fiber_dictionary[f_id] = list_fibers[f_id]
                        complete_fiber_set.add(f_id)
                        if(f_id in edge_fiber_set):
                            edge_fiber_set.remove(f_id)
                    else:
                        if(f_id not in complete_fiber_set):
                            edge_fiber_set.add(f_id)
                            fiber_dictionary[f_id] = list_fibers[f_id]
                Volume[x:x + window_size, y:y + window_size, z:z + window_size] = temp_vol.cpu().numpy()
                print("Found Complete Fibers: {}. Edge Fibers {} ".format(len(complete_fiber_set), len(edge_fiber_set)))
                counter = counter + 1

                f = open(h5_volume_dir + "/fiber_dictionary.txt", "w")
                for k in fiber_dictionary.keys():
                    el = fiber_dictionary[k]
                    f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.0f},{:.0f},{:.2f}\n".format(el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7], el[8]))
                f.close()
                del temp_vol
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    print("Saving Partition")
    tensors_io.save_volume_h5(Volume, name=dataset_name, dataset_name=dataset_name, directory=h5_volume_dir)
    print("Fitting Very Long Fibers")
    Volume = torch.from_numpy(Volume)
    list_fibers = get_fiber_properties_post_processing(Volume, offset=[0, 0, 0], complete_fibers=complete_fiber_set, edge_fiber_set={}, fibers_to_label=edge_fiber_set)

    for f_id in list_fibers.keys():
        fiber_dictionary[f_id] = list_fibers[f_id]

    f = open(h5_volume_dir + "/fiber_dictionary.txt", "w")
    for k in fiber_dictionary.keys():
        el = fiber_dictionary[k]
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.0f},{:.0f},{:.2f}\n".format(el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7], el[8]))
    f.close()


def find_n(Volume, neighbor_dictionary, neighbor_overlap):
    indices = np.unique(Volume)
    indices_set = set(indices)
    indices_set.remove(0)
    if(1 in indices_set):
        indices_set.remove(1)
    # se = ball(3)
    for idx in indices:
        if(idx == 1 or idx == 0):
            continue
        temp_set = set()
        temp_set = temp_set.union(indices_set)
        temp_set.remove(idx)
        if(len(temp_set) > 0):
            if(idx in neighbor_dictionary.keys()):
                neighbor_dictionary[idx] = neighbor_dictionary[idx].union(temp_set)
            else:
                neighbor_dictionary[idx] = temp_set
        '''
        temp_vl = np.zeros(Volume.shape)
        temp_vl[np.where(Volume == idx)] = 1
        temp_vl = binary_dilation(temp_vl, se)

        overlap_ids = np.unique(temp_vl * Volume)
        for i_ovlp in overlap_ids:
            if(i_ovlp != idx and i_ovlp > 1):
                neighbor_dictionary[idx][2].add(i_ovlp)
        '''

def find_neighboring_fibers(h5_volume_dir, dataset_name, window_size=50):
    print("Cropping volume")
    neighbor_dictionary = defaultdict(set)
    neighbor_overlap = []
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    rows, cols, slices = Volume.shape
    print(Volume.shape)
    start_x = []
    start_y = []
    start_z = []

    sz = 0
    while(sz + window_size < rows):
        start_x.append(sz)
        sz = sz + window_size

    sz = 0
    while(sz + window_size < cols):
        start_y.append(sz)
        sz = sz + window_size

    sz = 0
    while(sz + window_size < slices):
        start_z.append(sz)
        sz = sz + window_size

    num_partitions = len(start_z) * len(start_y) * len(start_x)
    counter = 0
    print("Starting Fitting")
    for x in start_x:
        for y in start_y:
            for z in start_z:
                print("Partition {} done out of {}".format(counter, num_partitions))
                temp_vol = Volume[x:x + window_size, y:y + window_size, z:z + window_size]
                find_n(temp_vol, neighbor_dictionary, neighbor_overlap)

                # np.save("neighbor_dictionary", neighbor_dictionary)
                counter += 1

    f = open( "./neighbor_elements2.txt", "w")
    for k in neighbor_dictionary.keys():
        el = neighbor_dictionary[k]
        f.write("{},".format(k))
        for it in el:
            f.write("{},".format(it))
        f.write("\n")
    f.close()

def crop_at_a_specific_fiber(h5_volume_dir, dataset_name, fiber_id=2):
    print("Cropping volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    coords = np.where(Volume == fiber_id)
    centers = [int(coords[i].mean()) for i in range(3)]
    maxs = [int(coords[i].max()) for i in range(3)]
    mins = [int(coords[i].min()) for i in range(3)]

    distances = [maxs[i] - mins[i] for i in range(3)]
    window_size = [distances[i] for i in range(3)]
    min_coords_xyz = [int(max(0, mins[i] - 20)) for i in range(3)]
    max_coords_xyz = [int(min(Volume.shape[i], maxs[i] + 20)) for i in range(3)]

    Volume = torch.from_numpy(Volume)
    properties = get_fiber_properties_post_processing(Volume, offset=[0, 0, 0], complete_fibers={}, fibers_to_label={fiber_id})
    Volume = Volume.numpy()
    print(properties)
    print("Window Size: {}".format(window_size))
    print(fiber_id, centers)
    Volume = Volume[min_coords_xyz[0]:max_coords_xyz[0], min_coords_xyz[1]:max_coords_xyz[1], min_coords_xyz[2]:max_coords_xyz[2]]
    # idx = np.where(Volume == fiber_id)
    # Volume = Volume * 0
    # Volume[idx] = 1000000
    tensors_io.save_volume_h5(Volume, name=dataset_name + "_cropped_single", dataset_name=dataset_name + "_cropped_single", directory=h5_volume_dir)

def crop_at_a_specific_fiber_neighbors(h5_volume_dir, dataset_name, neighbors_dir):
    print("Cropping volume")
    neighbors = read_neighbor_dictionary(neighbors_dir)

    fiber_dict = read_fiber_dictionary('h5_statistics/fiber_dictionary.txt')
    void_dictionary = read_fiber_dictionary('h5_statistics/void_dictionary2.txt')

    for k in neighbors.keys():
        if k > 800:
            fiber_id = k
            list_of_neighbors = neighbors[k]
            break

    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    coords = np.where(Volume == fiber_id)
    centers = [int(coords[i].mean()) for i in range(3)]
    maxs = [int(coords[i].max()) for i in range(3)]
    mins = [int(coords[i].min()) for i in range(3)]

    distances = [maxs[i] - mins[i] for i in range(3)]
    window_size = [distances[i] for i in range(3)]
    min_coords_xyz = [int(max(0, mins[i] - 20)) for i in range(3)]
    max_coords_xyz = [int(min(Volume.shape[i], maxs[i] + 20)) for i in range(3)]


    Volume = Volume[min_coords_xyz[0]:max_coords_xyz[0], min_coords_xyz[1]:max_coords_xyz[1], min_coords_xyz[2]:max_coords_xyz[2]]
    Volume2 = np.zeros(Volume.shape)

    list_of_neighbors.append(fiber_id)
    void_list = []
    fiber_list = []
    for el in list_of_neighbors:
        idx = np.where(Volume == el)
        if(el == fiber_id):
            Volume2[idx] = 1
        elif(el > 1000000):
            Volume2[idx] = 2
            print("Void properties")
            if(el in void_dictionary.keys()):
                void_list.append(void_dictionary[el])
        else:
            Volume2[idx] = 3
            fiber_list.append(fiber_dict[el])

    print("Fiber properties")
    print(fiber_dict[fiber_id])
    '''
    for i in fiber_list:
        print(i)

    print("Viods")
    for i in void_list:
        print(i)
    '''
    fiber_list = np.array(fiber_list)
    void_list = np.array(void_list)

    lengths = fiber_list[:, 5]
    thetas = fiber_list[:, 6]
    phis = fiber_list[:, 7]


    for i in lengths:
        print(i, end=",")
    print("")
    for i in thetas:
        print(i, end=",")
    print("")
    for i in phis:
        print(i, end=",")
    print("")
    print(lengths.mean())
    print(thetas.mean())
    print(phis.mean())
    print(void_list[:, 5].mean())

    direction = np.array([void_list[:, 6].mean(), void_list[:, 7].mean(), void_list[:, 8].mean()])
    Txy = np.arctan2(direction[1], direction[0]) * 180 / np.pi
    if(Txy < 0):
        Txy = 180 + Txy
    Tz = np.arccos(np.dot(direction, np.array([0, 0, 1])) / np.linalg.norm(direction, 2)) * 180 / np.pi
    print(Txy)
    print(Tz)
    tensors_io.save_volume_h5(Volume2, name=dataset_name + "_cropped_single", dataset_name=dataset_name + "_cropped_single", directory=h5_volume_dir)




def crop_at_a_specific_fiber_debug(h5_volume_dir, dataset_name, fiber_id=2):
    print("Cropping volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    coords = np.where(Volume == fiber_id)
    centers = [int(coords[i].mean()) for i in range(3)]
    maxs = [int(coords[i].max()) for i in range(3)]
    mins = [int(coords[i].min()) for i in range(3)]

    distances = [maxs[i] - mins[i] for i in range(3)]
    window_size = [1.2 * distances[i] for i in range(3)]
    min_coords_xyz = [int(max(0, mins[i])) for i in range(3)]
    max_coords_xyz = [int(min(Volume.shape[i], maxs[i])) for i in range(3)]

    print("Window Size: {}".format(window_size))
    print(fiber_id, centers)
    Volume = Volume[min_coords_xyz[0]:max_coords_xyz[0], min_coords_xyz[1]:max_coords_xyz[1], min_coords_xyz[2]:max_coords_xyz[2]]

    Volume_wrong = np.zeros(Volume.shape)
    Volume_small = np.zeros(Volume.shape)

    print("Fitting")
    list_fibers = fit_all_fibers_parallel(Volume)
    for k in list_fibers:
        if(k[0] == fiber_id):
            fiber_selected = k
        if(k[-1] > 200):
            Volume_wrong[np.where(Volume == k[0])] = k[0]
            print(k)

        if(k[-1] == -1):
            Volume_small[np.where(Volume == k[0])] = k[0]

    print(fiber_selected)
    tensors_io.save_volume_h5(Volume, name=dataset_name + "_cropped_single", dataset_name=dataset_name + "_cropped_single", directory=h5_volume_dir)
    tensors_io.save_volume_h5(Volume_wrong, name=dataset_name + "_wrong_fibers", dataset_name=dataset_name + "_wrong_fibers", directory=h5_volume_dir)
    tensors_io.save_volume_h5(Volume_small, name=dataset_name + "_small_fibers", dataset_name=dataset_name + "_wrong_fibers", directory=h5_volume_dir)


''' Crops a Volume given start coordinates and a window size'''
def crop_volume(h5_volume_dir, dataset_name, dataset_name2, start=[600, 600, 80], window_size=100):
    print("Cropping volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    print(len(np.where(Volume == 1)[0]))
    Volume = Volume[start[0]:start[0] + window_size,start[1]:start[1] + window_size, start[2]:start[2] + window_size]
    
    data_volume = tensors_io.read_volume_h5(dataset_name2, dataset_name2, h5_volume_dir)
    data_volume = data_volume[start[0]:start[0] + window_size,start[1]:start[1] + window_size,start[2]:start[2] + window_size]

    tensors_io.save_volume_h5(Volume, name=dataset_name + "_cropped", dataset_name=dataset_name + "_cropped", directory=h5_volume_dir)
    tensors_io.save_volume_h5((data_volume * 256).astype(np.int16), name= "volume_cropped", dataset_name= "volume_cropped", directory=h5_volume_dir)


''' Downsamples a volume given a scale'''
def downsample_volume(h5_volume_dir, dataset_name, scale=2):
    scale = float(scale)
    print("Downsampling volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    print(Volume.shape)
    Volume = ndi.zoom(Volume, 1.0 / scale, order=0)
    tensors_io.save_volume_h5(Volume, name=dataset_name + "_small", dataset_name=dataset_name + "_small", directory=h5_volume_dir)


''' Downsamples a volume given a scale'''
def upsample_full_volume(h5_volume_dir, dataset_name, scale=2):
    print("Upsampling Volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    slices = Volume.shape[-1]

    for slc in range(0, slices, 100):
        Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
        Volume = Volume[..., slc: slc + 100]
        Volume = ndi.zoom(Volume, scale, order=0)

        if(slc == 0):
            tensors_io.save_volume_h5(Volume, directory=h5_volume_dir, name=dataset_name + "_full_resolution", dataset_name=dataset_name + "_full_resolution")
        else:
            tensors_io.append_volume_h5(Volume, directory=h5_volume_dir, name=dataset_name + "_full_resolution", dataset_name=dataset_name + "_full_resolution")



''' Converts a Volume into a Volume represented by its size'''
def get_size_image(volume):
    print("Starting Labeling")
    volume, num_voids = measure.label(volume, return_num=True)
    print(num_voids)
    print("Starting Mapping")
    for i in range(1, num_voids):
        idx = np.where(volume == i)
        volume[idx] = len(idx[0])
    return volume

################################  Get Statistics ################################

''' Calculate Void Point Statistics'''
''' Outputs a Dictionary of the Form:'''
'''  << void_label, center[0]. center[1], center[2], mean_radious, volume, direction[0], direction[1], direction[2]'''
def get_statistics_voids(h5_volume_dir, dataset_name, scale=2):
    scale = float(scale)
    print("Calculating statistics voids")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    Volume[np.where(Volume != 1)] = 0

    Volume = ndi.zoom(Volume, 1.0 / scale)
    Volume, num_voids = measure.label(Volume, return_num=True)
    Volume[np.where(Volume > 0)] += 1000000

    list_voids = fit_all_voids_parallel(Volume)
    for el in list_voids:
        f = open(h5_volume_dir + "/void_dictionary.txt", "w")
        for k in range(len(list_voids)):
            el = list_voids[k]
            if(el[-1] != -1):
                f.write("{},{:.0f},{:.0f},{:.0f},{:.2f}, {:.3f}, {:.4f}, {:.4f}, {:.4f}\n".format(el[0], int(scale) * el[1].item(), int(scale) * el[2].item(), int(scale) * el[3].item(), int(scale) * el[4].item(), int(scale) * int(scale) * int(scale) * el[5], el[6], el[7], el[8]))
        f.close()
    Volume = ndi.zoom(Volume, 2, order=0)

    Volume = Volume[:-1, :-1, :-1]
    Volume_fibers = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    Volume_fibers[np.where(Volume_fibers == 1)] = Volume[np.where(Volume_fibers == 1)]

    tensors_io.save_volume_h5(Volume_fibers, name=dataset_name + '_labeled_voids', dataset_name=dataset_name + '_labeled_voids', directory=h5_volume_dir)

    try:
        read_dictionary_voids_volume(h5_volume_dir + "/void_statistics.txt")
    except:
        print("Volume plotting is not possible. Plot Manually")


''' Calculate Fiber Point Statistics'''
''' Outputs a Dictionary of the Form:'''
'''  << void_label, center[0]. center[1], center[2], mean_radious, length, theta, phi'''
def get_statistics_fibers(h5_volume_dir, fibers_name, scale=2):
    print("Downsampling and calculating statistics voids")
    scale = float(scale)
    V_fibers = tensors_io.read_volume_h5(fibers_name, fibers_name, h5_volume_dir)
    V_fibers[np.where(V_fibers == 1)] = 0

    V_fibers = ndi.zoom(V_fibers, 1.0 / scale, order=0)

    list_fibers = fit_all_fibers_parallel(V_fibers)
    f = open("dict_fibers.txt","w")
    for k in range(len(list_fibers)):
        el = list_fibers[k]
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.0f},{:.0f}\n".format(el[0], scale * el[1][0], scale * el[1][1], scale * el[1][2], scale * el[2], scale * el[3], el[4], el[5]))
    f.close()

################################ Plotting ################################
''' Read Fiber Dictionary and Plot Length Histogram'''
def plot_dictionary(directory, directory_voids=None, neighbors=None):
    fiber_dict = read_fiber_dictionary(directory)
    if(directory_voids is not None):
        void_dictionary = read_fiber_dictionary(directory_voids)

    if(neighbors is not None):
        neighbors = read_neighbor_dictionary(neighbors)  
    list_of_fibers = []
    for k in fiber_dict.keys():
        list_of_fibers.append(fiber_dict[k])

    list_of_fibers = np.array(list_of_fibers)
    num_fibers = len(list_of_fibers)
    print("Plotting for {} fibers".format(num_fibers))
    lenghts = 1.3 * np.clip(list_of_fibers[:, 5], 0, 1000000)

    radious = list_of_fibers[:, 4]
    thetas = list_of_fibers[:, 6]
    phis = list_of_fibers[:, 7]
    ratios = 5
    thetas = thetas[np.where(lenghts > ratios * radious)] - 90
    phis = phis[np.where(lenghts > ratios * radious)] - 90
    lenghts = lenghts[np.where(lenghts >ratios * radious)]
    (lenghts_histogram, bin_edges) = np.histogram(lenghts, 800)

    thetas = thetas.astype(np.float)
    phis = phis.astype(np.float)

    thetas = thetas * np.pi / 180.0
    phis = phis * np.pi / 180.0
    print(lenghts.mean())
    print(bin_edges.max())


    A11 = np.sin(thetas)**2 * np.cos(phis)**2
    A21 = np.sin(thetas)**2 * np.cos(phis) * np.sin(phis)
    A31 = np.sin(thetas) * np.cos(thetas) * np.cos(phis)
    print(A11)
    A12 = np.sin(thetas)**2 * np.cos(phis) * np.sin(phis)
    A22 = np.sin(thetas)**2 * np.sin(phis)**2
    A32 = np.sin(thetas) * np.cos(thetas) * np.sin(phis)

    A13 = np.sin(thetas) * np.cos(thetas) * np.cos(phis)
    A23 = np.sin(thetas) * np.cos(thetas) * np.sin(phis)
    A33 = np.cos(thetas)**2

    Aij = np.array([[A11.mean(), A12.mean(), A13.mean()], [A21.mean(), A22.mean(), A23.mean()], [A31.mean(), A32.mean(), A33.mean()]])

    print("HERE")
    print(np.arccos(A11.mean()) * 180 / np.pi)
    print(np.arccos(A22.mean()) * 180 / np.pi)
    print(np.arccos(A33.mean()) * 180 / np.pi)
    print("Here")
    print(Aij)

    plt.bar(bin_edges[:-1], lenghts_histogram.astype(np.float), width=1)
    plt.ylabel('Frequency')
    plt.xlabel('Lengths (um)')
    # plt.title(directory)
    #plt.title("Fiber Length Distribution")
    print("Plotting")
    plt.savefig(directory[:-4] + ".lenghts_histogram.png")
    #plt.show()
    exit()
    return list_of_fibers

def read_fiber_dictionary(directory):
    counter = 0
    fiber_dict = {}
    with open(directory) as f:
        for line in f.readlines():
            line = line[:-2].split(",")
            fiber_id = int(line[0])
            for k in line:
                try:
                    fiber_dict[fiber_id] = np.array([float(k) for k in line])
                except:
                    continue
            counter += 1
    
    return fiber_dict


def read_neighbor_dictionary(directory):
    print("Readng neighbors")
    counter = 0
    fiber_dict = {}
    with open(directory) as f:
        for line in f.readlines():
            line = line[:-2].split(",")
            fiber_id = int(line[0])
            for k in line:
                list_of_neighbors = [int(k) for k in line]
            fiber_dict[fiber_id] = list_of_neighbors[1:]
            counter += 1
            if(counter > 1000):
                break
    return fiber_dict
''' Read Void Dictionary and Plot Volume Histogram'''
def read_dictionary_voids_volume(directory, large_vol_dim=5000):
    scale = 8
    counter = 0
    list_of_voids = []
    with open(directory) as f:
        for line in f.readlines():
            line = line[:-1].split(",")
            list_of_voids.append([float(k) for k in line])
            counter += 1

    list_of_voids = np.array(list_of_voids)
    len(list_of_voids)

    volumes = list_of_voids[:, 5]
    volumes = np.delete(volumes, np.where(volumes < large_vol_dim))
    (volumes_histogram, bin_edges) = np.histogram(volumes, 100)

    fig = plt.figure()
    plt.plot(scale * bin_edges[:-1] * 1.3, volumes_histogram)
    plt.ylabel('Counts')
    plt.xlabel('Volume')
    plt.title('Large Void Volume Histogram')
    print("Plotting")
    plt.savefig(directory[:-4] + "_volumes.png")
    plt.close(fig)

    return list_of_voids


def get_fiber_properties_post_processing(space_labels, offset=[0, 0, 0], complete_fibers={}, edge_fiber_set={}, fibers_to_label=None, vol_dim=[100000, 100000, 100000]):
    end_points = []
    fiber_ids = []
    fiber_list = {}
    device = space_labels.device

    window_size = space_labels.shape[-1]

    if(fibers_to_label is None):
        fibers_to_label = torch.unique(space_labels)
    else:
        fibers_to_label = torch.tensor(list(fibers_to_label))

    num_fibers = len(fibers_to_label)
    counter = 0
    for fiber_id in fibers_to_label:
        edge_fiber = 0
        if(fiber_id.cpu().item() == 0 or fiber_id.cpu().item() == 1 or fiber_id.cpu().item() in complete_fibers):
            continue

        idx = (space_labels == fiber_id).nonzero().float()
        if(len(idx) == 0):
            continue
        # idx is shape [N, 3]

        center = idx.mean(0)
        rs0 = torch.norm(idx - center, p=2, dim=1)

        # Find farthest distance from center
        end_point0_idx = (rs0 == rs0.max()).nonzero()
        
        # Find points close to EP1
        end_point0_idx = end_point0_idx[0, 0]
        idx_split = idx.split(1, dim=1)

        end_point0 = torch.tensor([idx_split[0][end_point0_idx], idx_split[1][end_point0_idx], idx_split[2][end_point0_idx]], device=idx.device)
        
        # Find closes points from end point 0
        rs1 = torch.norm(idx - end_point0, p=2, dim=1)
        end_point1_idx = (rs1 < 3).nonzero()
        end_point1_idx = end_point1_idx[:, 0]
        end_point1 = torch.tensor([idx_split[i][end_point1_idx][:, 0].mean() for i in range(3)])
         #, idx_split[1][end_point1_idx], idx_split[2][end_point1_idx]], device=idx.device)
     
        # Find farthest point from end point 1
        rs2 = torch.norm(idx - end_point0, p=2, dim=1)
        # Find farthest point from end point 1
        end_point2_idx = (rs2 > rs2.max() - 3).nonzero()
        end_point2_idx = end_point2_idx[:, 0]
        end_point2 = torch.tensor([idx_split[i][end_point2_idx][:, 0].mean() for i in range(3)])

        end_points.append(end_point1.cpu().numpy())
        end_points.append(end_point2.cpu().numpy())

        fiber_ids.append(fiber_id.cpu().item())
        fiber_ids.append(fiber_id.cpu().item())

        c_np = center.cpu().numpy()

        length = torch.norm(end_point1 - end_point2, p=2).cpu().item()
        direction = (end_point1 - end_point2)
        direction = direction / torch.norm(direction, p=2)

        # Find Fibers in the Edge of Overlapping Volume
        for i in range(3):
            # Offset must be > 0 to be a real edge Fiber
            if((end_point1[i] < 5 or end_point2[i] < 5) and offset[i] > 0):
                edge_fiber = 1
                break

            # Here would be good to find the offset value to know when I am in the other edge
            if((end_point1[i] > space_labels.shape[i] - 6 or end_point2[i] > space_labels.shape[i] - 6) and (offset[i] + window_size < vol_dim[i] - 10)):
                edge_fiber = 1
                break

        n = idx.shape[0]
        step_size = 1
        if(n > 30):
            step_size = n / 30
        if((n < 10 or math.isnan(direction[0])) and edge_fiber == 0):
            idx2 = (space_labels == fiber_id).nonzero()
            space_labels[idx2.split(1, dim=1)] = 0
            if(fiber_id.cpu().item() in edge_fiber_set):
                edge_fiber_set.remove(fiber_id.cpu().item())
            continue

        rr = fit_t.r(direction.unsqueeze(1).to(device), idx, center.unsqueeze(1))
        rr = rr.cpu().item()

        idx = idx[::step_size, :]
        G = fit_t.G(direction.unsqueeze(1).to(device), idx, center.unsqueeze(1))
        G = G.cpu().item()

        direction = direction.cpu().numpy()

        Txy = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        if(Txy < 0):
            Txy = 180 + Txy
        Tz = np.arccos(np.dot(direction, np.array([0, 0, 1])) / np.linalg.norm(direction, 2)) * 180 / np.pi

        fiber_list[fiber_id.cpu().item()] = [fiber_id.cpu().item(), 2 * (c_np[0] + offset[0]), 2 * (c_np[1] + offset[1]), 2 * (c_np[2] + offset[2]), 2 * rr, 2 * length, Txy, Tz, G, edge_fiber]

        if(counter % int(num_fibers / 10) == 0):
            print("{} done".format(int(counter * 100) / num_fibers + 1))
        counter += 1

    return fiber_list


def fill_watershed(labels, sample_vol):
    segmentation = np.zeros(sample_vol.shape)
    segmentation[np.where(sample_vol == 1)] = 1

    markers = np.copy(labels)

    distance = ndi.distance_transform_edt(segmentation)
    distance[np.where(labels > 0)] = 1
    labels = watershed(-distance, markers, mask=segmentation)

    return labels

def read_nearby_objects(path):
    dictionary = np.load(path).item()
    f = open( "./neighbor_elements.txt", "w")
    for k in dictionary.keys():
        el = dictionary[k]
        f.write("{},".format(k))
        for it in el:
            f.write("{},".format(it))
        f.write("\n")
    f.close()


def process_neighbor_dicts(directory, directory_voids=None, neighbors=None):
    print("processing fibers")
    fiber_dict = read_fiber_dictionary(directory)
    if(directory_voids is not None):
        void_dictionary = read_fiber_dictionary(directory_voids)

    if(neighbors is not None):
        neighbors = read_neighbor_dictionary(neighbors) 
    list_of_fibers = []
    for k in fiber_dict.keys():
        list_of_fibers.append(fiber_dict[k])

    list_of_fibers = np.array(list_of_fibers)
    num_fibers = len(list_of_fibers)
    lenghts = 1.3 * np.clip(list_of_fibers[:, 5], 0, 1000000)

    radious = list_of_fibers[:, 4]
    thetas = list_of_fibers[:, 6]
    phis = list_of_fibers[:, 7]
    ratios = 1
    thetas = thetas[np.where(lenghts > ratios * radious)] - 90
    phis = phis[np.where(lenghts > ratios * radious)] - 90
    lenghts = lenghts[np.where(lenghts >ratios * radious)]

    neighbor_dict = {}
    for fiber in neighbors.keys():
        thetas_f = []
        thetas_v = []
        for nearby_el in neighbors[fiber]:
            if nearby_el in fiber_dict.keys():
                thetas_f.append(fiber_dict[nearby_el][6])
            if nearby_el in void_dictionary.keys():
                thetas_v.append(void_dictionary[nearby_el][6])
        neighbor_dict[fiber] = (thetas_f, thetas_v)

    print(neighbor_dict)
    return list_of_fibers
if __name__ == '__main__':
    data_path = '/Storage/DATASETS/Fibers/Tiff_files_tomo_data'
    data_path = "/pub2/aguilarh/DATASETS/Tiff_files_tomo_data"
    
    ########################  Downsample Volumes  ########################
    downsample_volume("h5_statistics", "volume_fiber_voids_labeled_voids")
    # crop_volume("output_files", "volume_fiber_voids", "volume_fiber_voids")
    #for i in range(1, 5):
    #    mask_path = "MORE_TRAINING/NewTrainData_Sep9/sV" + str(i) + "/fibers_uint16_sV" + str(i)
    #    get_direction_training(mask_path)
    # directions_plotting()
    # Getting Voids Momemtum
    #massive_fitting('h5_files',  volume_h5_name='final_fibers', start=0, end=100)
    # get_statistics_voids("statistics", "only_voids")
    #displaying_statistics('statistics', 'final_fibers', 'voids')   
    
    ######################## Dictionary fibers whole volume ########################
    # FIBERS
    # read_dictionary("h5_statistics/fiber_dictionary.txt")
    start = time.time()
    # get_whole_volume_statistics("h5_statistics", "volume_fiber_voids")
    # fit_long_fibers("h5_statistics", "volume_fiber_voids")
    #find_neighboring_fibers("h5_statistics", "volume_fiber_voids_labeled_voids")
    # print(time.time() - start)
    # VOIDS
    #read_dictionary_voids_volume("h5_statistics/dict_voids.txt")

    ######################## Saving Volumes ########################
    # tensors_io.save_images_of_h5(h5_volume_dir='output_files', data_volume_path=data_path, output_path='output_files/fibers', volume_h5_name='volume_fiber_voids', start=10, end=20, scale=2)
    
    # tensors_io.save_images_of_h5_side(h5_volume_dir='output_files', data_volume_path=data_path, output_path='output_files/fibers_side', volume_h5_name='volume_fiber_voids', start=400, end=410, scale=2)
    # This import registers the 3D projection, but is otherwise unused.
    
    # read_nearby_objects('neighbor_dictionary.npy')
    #crop_at_a_specific_fiber("h5_statistics", "volume_fiber_voids_labeled_voids", 197)
    #crop_at_a_specific_fiber_neighbors("h5_statistics", "volume_fiber_voids_labeled_voids", 'neighbor_elements2.txt')
    # fiber_dict = plot_dictionary('h5_statistics/fiber_dictionary.txt', 'h5_statistics/void_dictionary2.txt', 'neighbor_elements2.txt')
    #process_neighbor_dicts('h5_statistics/fiber_dictionary.txt', 'h5_statistics/void_dictionary2.txt', 'neighbor_elements2.txt')
    #read_dictionary_voids_volume('h5_statistics/void_dictionary2.txt')
    
    #print(len(fiber_dict.keys()))
    #print(fiber_dict)
