import time
import torch
from unet import UNet
from fiber_void_fns import process_all_volume

if __name__ == '__main__':
    ################################# PATH TO TIFF FILES TOMO ##########################
    data_path_f0 = '/Storage/DATASETS/Fibers/Tiff_files_tomo_data'
    data_path_f = "/pub2/aguilarh/DATASETS/Tiff_files_tomo_data"

    #######################################################################
    sub_volume_size = 384
    device = torch.device("cuda:0")

    ######################### CNNs Definition ########################################
    net_fibers1 = UNet(n_channels=1, n_classes=2, num_dims=64)
    net_fibers2 = UNet(n_channels=1, n_classes=12, num_dims=64)
    net_voids = UNet(n_channels=1, n_classes=3, num_dims=10)
    ##################################################################################

    start = time.time()


    process_all_volume(net_fibers1, net_fibers2, net_voids, data_path_f, n_embedded=12, cube_size_e=64, sub_volume_size=sub_volume_size, n_classes=2, scale=2, device=device)
    # process_all_volume_fibers(net_fibers1, net_fibers2, data_path_f, n_embedded=12, cube_size_e=64, sub_volume_size=sub_volume_size, n_classes=2, scale=2, device=device)
    # process_all_volume_voids(net_voids, data_path_f, n_embedded=12, cube_size_e=64, sub_volume_size=384, n_classes=3, scale=2, device=device)
    end = time.time()
    print(end - start)
