import time
import torch
from unet import UNet
from fiber_void_fns import process_all_volume
from statistics import get_statistics_voids, read_dictionary_voids_volume, upsample_full_volume
import tensors_io

if __name__ == '__main__':
    ################################# PATH TO TIFF FILES TOMO ##########################
    data_path_f0 = '/Storage/DATASETS/Fibers/Tiff_files_tomo_data'
    data_path_f = "/pub2/aguilarh/DATASETS/Tiff_files_tomo_data"

    output_folder_name = "./output_files"

    #######################################################################
    sub_volume_size = 384
    device = torch.device("cuda:0")

    ######################### CNNs Definition ########################################
    net_fibers1 = UNet(n_channels=1, n_classes=2, num_dims=64)
    net_fibers2 = UNet(n_channels=1, n_classes=12, num_dims=64)
    net_voids = UNet(n_channels=1, n_classes=3, num_dims=10)
    ##################################################################################

    start = time.time()

    # Process Voids and Fibers
    process_all_volume(net_fibers1, net_fibers2, net_voids, data_path_f, n_embedded=12, cube_size_e=64, sub_volume_size=sub_volume_size, n_classes=2, scale=2, device=device, output_directory=output_folder_name)

    end = time.time()
    print("Time taken to detect voids and fibers: {}".format(end - start))

    # Get Voids Statistics
    get_statistics_voids(output_folder_name, "volume_fiber_voids", scale=2)
    
    # Upsample Volume
    upsample_full_volume(output_folder_name, "volume_fiber_voids", scale=1)


    print("Saving Sample Outputs")
    tensors_io.save_images_of_h5(h5_volume_dir=output_folder_name, data_volume_path=data_path_f, output_path='output_files/fibers_full', volume_h5_name='volume_fiber_voidsfull', start=600, end=610, scale=1)
    tensors_io.save_images_of_h5_side(h5_volume_dir=output_folder_name, data_volume_path=data_path_f, output_path='output_files/fibers_side', volume_h5_name='volume_fiber_voidsfull', start=1000, end=1010, scale=1)

   