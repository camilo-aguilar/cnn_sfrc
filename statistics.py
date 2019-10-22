import tensors_io
from skimage import measure
from unet.cylinder_fitting import fit_all_fibers_parallel, fit_all_voids_parallel
import numpy as np

import matplotlib.pyplot as plt
import scipy.ndimage as ndi



################################  Manipulate Volumes ################################
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
def downsample_volume(h5_volume_dir, dataset_name, scale=4):
    scale = float(scale)
    print("Downsampling volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)

    Volume = ndi.zoom(Volume, 1 / scale, order=0)
    tensors_io.save_volume_h5(Volume, name=dataset_name + "_small", dataset_name=dataset_name + "_small", directory=h5_volume_dir)


''' Downsamples a volume given a scale'''
def upsample_full_volume(h5_volume_dir, dataset_name, scale=2):
    print("Upsampling Volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    slices = Volume.shape[-1]
    upsample_full_volume

    for slc in range(0, slices, 100):
        Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
        Volume = Volume[..., slc: slc + 100]
        Volume = ndi.zoom(Volume, scale, order=0)

        if(slc == 0):
            tensors_io.save_volume_h5(Volume, directory=h5_volume_dir, name=dataset_name + "full", dataset_name=dataset_name + "_full")
        else:
            tensors_io.append_volume_h5(Volume, directory=h5_volume_dir, name=dataset_name + "full", dataset_name=dataset_name + "_full")



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
def get_statistics_voids(h5_volume_dir, voids_name, scale=2):
    scale = float(scale)
    print("Calculating statistics voids")
    Volume = tensors_io.read_volume_h5(voids_name, voids_name, h5_volume_dir)
    Volume[np.where(Volume != 1)] = 0

    Volume = ndi.zoom(Volume, 1.0 / scale)
    Volume, num_voids = measure.label(Volume, return_num=True)
    list_voids = fit_all_voids_parallel(Volume)
    for el in list_voids:
        f = open(h5_volume_dir + "/void_dictionary.txt", "w")
        for k in range(len(list_voids)):
            el = list_voids[k]
            if(el[-1] != -1):
                f.write("{},{:.0f},{:.0f},{:.0f},{:.2f}, {:.3f}, {:.4f}, {:.4f}, {:.4f}\n".format(el[0], int(scale) * el[1].item(), int(scale) * el[2].item(), int(scale) * el[3].item(), int(scale) * el[4].item(), int(scale) * int(scale) * int(scale) * el[5], el[6], el[7], el[8]))
        f.close()
    Volume = ndi.zoom(Volume, 2, order=0)
    tensors_io.save_volume_h5(Volume, name=voids_name + '_labeled_voids', dataset_name=voids_name + '_labeled_voids', directory=h5_volume_dir)

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

    V_fibers = ndi.zoom(V_fibers, 1 / scale, order=0)

    list_fibers = fit_all_fibers_parallel(V_fibers)
    f = open("dict_fibers.txt","w")
    for k in range(len(list_fibers)):
        el = list_fibers[k]
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.0f},{:.0f}\n".format(el[0], scale * el[1][0], scale * el[1][1], scale * el[1][2], scale * el[2], scale * el[3], el[4], el[5]))
    f.close()

################################ Plotting ################################
''' Read Fiber Dictionary and Plot Length Histogram'''
def read_dictionary(directory):
    counter = 0
    list_of_fibers = []
    with open(directory) as f:
        for line in f.readlines():
            line = line[:-2].split(",")
            for k in line:
                try:
                    list_of_fibers.append([float(k) for k in line])
                except:
                    continue
            counter += 1
    list_of_fibers = np.array(list_of_fibers)
    len(list_of_fibers)

    lenghts = np.clip(list_of_fibers[:, 5], 0, 1000)
    (lenghts_histogram, bin_edges) = np.histogram(lenghts, 100)


    plt.plot(bin_edges[:-1], lenghts_histogram)
    plt.ylabel('Counts')
    plt.xlabel('Lenghts')
    print("Plotting")
    plt.show()

    return list_of_fibers

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


if __name__ == '__main__':
    data_path = '/Storage/DATASETS/Fibers/Tiff_files_tomo_data'
    data_path = "/pub2/aguilarh/DATASETS/Tiff_files_tomo_data"
    
    ########################  Downsample Volumes  ########################
    #downsample_volume("h5_files", "final_fibers")
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
    #read_dictionary("statistics/dict_cool.txt")
    #read_dictionary("/home/camilo/Desktop/development/instances/embedding/dict_single.txt")

    # VOIDS
    #read_dictionary_voids_lenghts("statistics/dict_voids.txt")

    ######################## Saving Volumes ########################
    # tensors_io.save_images_of_h5(h5_volume_dir='output_files', data_volume_path=data_path, output_path='output_files/fibers', volume_h5_name='volume_fiber_voids', start=10, end=20, scale=2)
    
    # tensors_io.save_images_of_h5_side(h5_volume_dir='output_files', data_volume_path=data_path, output_path='output_files/fibers_side', volume_h5_name='volume_fiber_voids', start=400, end=410, scale=2)
    # This import registers the 3D projection, but is otherwise unused.
    