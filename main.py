import time
import torch
from unet import UNet
from fiber_void_fns import process_all_volume
from statistics import get_statistics_voids, read_dictionary_voids_volume, upsample_full_volume, get_whole_volume_statistics
import tensors_io

if __name__ == '__main__':
    ################################# PATH TO TIFF FILES TOMO ##########################
    data_path_f0 = '/Storage/DATASETS/Fibers/Tiff_files_tomo_data'
    data_path_f = "/pub2/aguilarh/DATASETS/Tiff_files_tomo_data"

    output_folder_name = "./output_files"

    #######################################################################
    sub_volume_size = 192
    device = torch.device("cuda:0")

    ######################### CNNs Definition ########################################
    net_fibers1 = UNet(n_channels=1, n_classes=2, num_dims=64)
    net_fibers2 = UNet(n_channels=1, n_classes=12, num_dims=64)
    net_voids = UNet(n_channels=1, n_classes=3, num_dims=10)
    ##################################################################################

    start0 = time.time()

    # Process Voids and Fibers
    start = time.time()
    process_all_volume(net_fibers1, net_fibers2, net_voids, data_path_f, n_embedded=12, cube_size_e=64, sub_volume_size=sub_volume_size, n_classes=2, scale=2, device=device, output_directory=output_folder_name)
    end = time.time()
    print("Time taken to detect voids and fibers: {}".format(end - start))

    # Get Fiber Statistics
    start = time.time()
    get_whole_volume_statistics(output_folder_name, dataset_name="volume_fiber_voids", device=device)
    end = time.time()
    print("Time taken to get statistics: {}".format(end - start))

    # Get Voids Statistics
    get_statistics_voids(output_folder_name, dataset_name="volume_fiber_voids", scale=2)
    # OUTPUTS: volume_fiber_voids_labeled_voids.xmf

    # Upsample Volume
    upsample_full_volume(output_folder_name, dataset_name="volume_fiber_voids_labeled_voids", scale=2)

    end = time.time()
    print("Time taken to run code: {}".format(end - start0))

    ################################################ SAMPLE TIFF IMAGES FROM H5 VOLUMES #############################################
    # print("Saving Sample Images Outputs")
    # Sample slices to save with top view

    # start_top = 300
    # end_top = 305
    # tensors_io.save_images_of_h5(h5_volume_dir=output_folder_name, data_volume_path=data_path_f, output_path=output_folder_name + '/fibers_full', volume_h5_name='volume_fiber_voids_labeled_voids', start=start_top, end=end_top, scale=2)

    # start_side = 300
    # end_side = 305
    # Sample slices to save with side view
    # tensors_io.save_images_of_h5_side(h5_volume_dir=output_folder_name, data_volume_path=data_path_f, output_path=output_folder_name + '/fibers_side', volume_h5_name='volume_fiber_voids_labeled_voids', start=start_side, end=end_side, scale=2)
