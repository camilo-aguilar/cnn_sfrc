import tensors_io
from skimage import measure
from unet.cylinder_fitting import fit_all_fibers_parallel_simple, fit_all_fibers_parallel, fit_all_fibers_parallel_simple
import numpy as np
import matplotlib.pyplot as plt



def crop_volume(h5_volume_dir, dataset_name, dataset_name2, start=[600, 600, 80], end=100):
    start = [650, 450, 50]
    print("Cropping volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    print(len(np.where(Volume == 1)[0]))
    Volume = Volume[start[0]:start[0] + end,start[1]:start[1] + end, start[2]:start[2] + end]
    for i in np.unique(Volume):
        pixs = len(np.where(Volume == i)[0])
        if(pixs < 20):
            print(i)
    
    data_volume = tensors_io.read_volume_h5(dataset_name2, dataset_name2, h5_volume_dir)
    data_volume = data_volume[start[0]:start[0] + end,start[1]:start[1] + end,start[2]:start[2] + end]

    tensors_io.save_volume_h5(Volume, name=dataset_name + "_cropped", dataset_name=dataset_name + "_cropped", directory=h5_volume_dir)
    tensors_io.save_volume_h5((data_volume * 256).astype(np.int16), name= "volume_cropped", dataset_name= "volume_cropped", directory=h5_volume_dir)

def downsample_volume(h5_volume_dir, dataset_name):

    print("Downsampling volume")
    Volume = tensors_io.read_volume_h5(dataset_name, dataset_name, h5_volume_dir)
    scale = 4
    # Volume = Volume[1:100, 1:100, 1:200]
    Volume = Volume[::scale, ::scale, ::scale]
    tensors_io.save_volume_h5(Volume, name=dataset_name + "_small", dataset_name=dataset_name + "_small", directory=h5_volume_dir)

def get_statistics_voids(h5_volume_dir, voids_name):

    print("Downsampling and calculating statistics voids")
    V_voids = tensors_io.read_volume_h5(voids_name, voids_name, h5_volume_dir)
    V_voids = V_voids > 0
    V_voids, num_voids = measure.label(V_voids, return_num=True)
    list_voids = fit_all_fibers_parallel(V_voids)
    
    for el in list_voids:
        if(el[-1] > 0):
            print(el)
            '''
    f = open("dict_voids.txt","w")
    for k in range(len(list_voids)):
        el = list_voids[k]
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f}, {:.2f}, {:.0f}, {:.0f}\n".format(el[0], el[1], el[2], [3], el[4], el[5], el[6], el[7]))
        # L, C_fit, r_fit, h_fit, Txy, Tz, fit_err
    f.close()
    '''
    momemtum_y = np.zeros(V_voids.shape).astype(np.int)
    momemtum_z = np.zeros(V_voids.shape).astype(np.int)

    for el in list_voids:
        Ty = el[6]
        Tz = el[7]
        if(Ty == -1 or Tz == -1):
            continue
        idx = np.where(V_voids == el[0])
        momemtum_y[idx] = int(Ty / 60)
        momemtum_z[idx] = int(Tz / 60)

    tensors_io.save_volume_h5(momemtum_z, name=voids_name + "_angle_z", dataset_name=voids_name + "_angle_z", directory=h5_volume_dir)
    tensors_io.save_volume_h5(momemtum_y, name=voids_name + "_angle_y", dataset_name=voids_name + "_angle_y", directory=h5_volume_dir)


def get_direction_training(input_dir):
    V_fibers = tensors_io.load_volume_uint16(input_dir, scale=1)

    V_fibers = V_fibers[0, ...].numpy()
    Vshape = V_fibers.shape
    list_fibers = fit_all_fibers_parallel_simple(V_fibers)
    output_dir = input_dir + 'directions'
    training_data = np.zeros([3, Vshape[0], Vshape[1], Vshape[2]])

    for el in list_fibers:
        if(el[0] == -1):
            print("one fiber skipped")
            continue
        idx = np.where(V_fibers == el[0])
        for i in range(3):
            tuple_v = tuple([np.array(i), idx[0], idx[1], idx[2]])
            w_c = el[1][i]
            training_data[tuple_v]  = w_c


    np.save(output_dir, training_data)        

def get_statistics_fibers(h5_volume_dir, fibers_name):
    print("Downsampling and calculating statistics voids")
    V_fibers = tensors_io.read_volume_h5(fibers_name, fibers_name, h5_volume_dir)
    scale = 4
    V_fibers = V_fibers[::scale, ::scale, ::scale]

    list_voids = fit_all_fibers_parallel(V_fibers)
    f = open("dict_fibers.txt","w")
    for k in range(len(list_voids)):
        el = list_voids[k]
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f},{:.2f},{:.0f},{:.0f}\n".format(el[0], el[1][0], el[1][1], el[1][2], el[2], el[3], el[4], el[5]))
        # L, C_fit, r_fit, h_fit, Txy, Tz, fit_err
    f.close()


################################################## Statistics    ####################################    

def massive_fitting(h5_volume_dir, volume_h5_name='Volume', start=0, end=None):
    print("Starting Massive Fitting")
    Vf = tensors_io.read_volume_h5(volume_h5_name, volume_h5_name, h5_volume_dir)

    if(end is None):
        end = Vf.shape[-1]

    Vf = torch.from_numpy(Vf[:, :, start:end].astype(np.long)).unsqueeze(0)

    final_list_fibers = fit_all_fibers_parallel_simple(Vf)
    f = open("dict.txt","w")
    for k in range(len(final_list_fibers)):
        el = final_list_fibers[k]
        f.write("{},{:.0f},{:.0f},{:.0f},{:.2f}\n".format(el[0], el[1][0], el[1][1], el[1][2], el[2]))
    f.close()


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
    # plt.axis(bin_edges)

    print("Plotting")
    plt.show()

    return list_of_fibers

def read_dictionary_voids_lenghts(directory):
    scale = 8
    counter = 0
    list_of_fibers = []
    with open(directory) as f:
        for line in f.readlines():
            line = line[:-1].split(",")
            list_of_fibers.append([float(k) for k in line])
            counter += 1

    list_of_fibers = np.array(list_of_fibers)
    len(list_of_fibers)

    # lenghts = np.clip(list_of_fibers[:, 5], 0, 500)
    lenghts = list_of_fibers[:, 5]
    lenghts = np.delete(lenghts, np.where(list_of_fibers == 1))
    (lenghts_histogram, bin_edges) = np.histogram(lenghts, 100)


    plt.plot(scale * 2 * bin_edges[:-1] * 1.3, lenghts_histogram)
    plt.ylabel('Counts')
    plt.xlabel('Lenght (m E-6)')
    plt.title('Void Lenght Histogram')
    # plt.axis(bin_edges)

    print("Plotting")
    plt.show()

    return list_of_fibers

def get_size_image(volume):
    print("Starting Labeling")
    volume, num_voids = measure.label(volume, return_num=True)
    # save_volume_h5(V_voids, name='only_voids', dataset_name='only_voids', directory='./statistics')
    print(num_voids)
    print("Starting Mapping")
    for i in range(1, num_voids):
        idx = np.where(volume == i)
        volume[idx] = len(idx[0])
    return volume

'''
def get_size_fibers(volume):
	read_dictionary("from_meanpill/dict.txt")
    print("Starting Labeling")
    volume, num_fibers = measure.label(volume, return_num=True)
    # save_volume_h5(V_voids, name='only_voids', dataset_name='only_voids', directory='./statistics')
    print(num_fibers)
    print("Starting Mapping")
    for i in range(1, num_fibers):
        idx = np.where(volume == i)
        volume[idx] = len(idx[0])
    return volume
'''

def directions_plotting():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    import matplotlib.pyplot as plt
    import numpy as np

    #V = np.load("updated_fibers/directions_300.npy")
    #V = np.load("network_output_directions.npy")
    V = np.load("temp_save.npy")
    print(V.shape)
    S = 0
    sz = 20
    V = V[:, S:S + sz, S:S + sz, S:S + sz]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print(V.shape)
    '''
    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2))

    # Make the direction data for the arrows
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))

    print(u.shape)
    print(x.shape)
    '''
    scale = 2
    V = V[:, ::scale, ::scale, ::scale] 
    u = V[1, ...]
    v = V[0, ...]
    w = V[2, ...]
    # Make the grid
    x, y, z = np.meshgrid(np.arange(0, V.shape[1], 1),
                          np.arange(0, V.shape[2], 1),
                          np.arange(0, V.shape[3], 1))
    ax.quiver(x, y, z, u, v, w, length=1, normalize=True)

    plt.show()
if __name__ == '__main__':
    data_path = '/Storage/DATASETS/Fibers/Tiff_files_tomo_data'
    data_path = "/pub2/aguilarh/DATASETS/Tiff_files_tomo_data"
    
    ########################  Downsample Volumes  ########################
    #downsample_volume("h5_files", "final_fibers")
    crop_volume("h5_files", "final_fibers", "data_volume")
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
    #tensors_io.save_images_of_h5(h5_volume_dir='h5_files', data_volume_path=data_path, output_path='h5_files/fibers', volume_h5_name='final_fibers', start=10, end=20, scale=2)
    
    # tensors_io.save_images_of_h5_side(h5_volume_dir='h5_files', data_volume_path=data_path, output_path='h5_files/fibers_side', volume_h5_name='final_fibers', start=400, end=410, scale=2)
    # This import registers the 3D projection, but is otherwise unused.
    