# full assembly of the sub-parts to form the complete net

from unet_parts import *
import pylab
from torchvision import transforms
from sklearn.cluster import DBSCAN, MeanShift
from cylinder_fitting import fit_all_fibers_parallel, post_processing, guess_cylinder_parameters_indexes, r_individual, guess_cylinder_parameters_merged
# from tsne import tsne, pca
import os
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import torch
import time
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import find as find_sparse
import cylinder_fitting.fitting as fit
from collections import defaultdict
from skimage import measure
from skimage.morphology import watershed, binary_erosion, ball, binary_dilation, binary_opening
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math
'''
import torch.nn as nn
'''


####################################################################################################################

def embedded_geometric_loss_coords2(outputs, labels):
    delta_v = 1
    delta_d = 5
    centroid_pixels = outputs[0]
    offset_vectors = outputs[1] #[:, 0:3, ...]
    # sigma_output = outputs[1][:, 3, ...].unsqueeze(0)

    device = offset_vectors.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)
    
    # copy 3D labels
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything and make offset_vectors 3 dimensions at the end
    labels = labels.contiguous().view(-1)

    N_pixels = len(labels)
    #sigma_output = sigma_output.contiguous().view(-1)
    offset_vectors = offset_vectors.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # mu vectors
    mu_vector = torch.zeros(N_objects, 3).to(device)
    probs = torch.zeros(labels.shape).to(device)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    #sigma_pixels = sigma_output[idx_tuple].squeeze(1)
    probs_results = torch.zeros_like(labeled_pixels).float()
    
    

    object_pixels = torch.gather(offset_vectors, 0, idx_array.repeat(1, 3))
    offset_loss = 0
    loss_mu = 0
    # sigma_loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]
        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        Nc = len(idx_c)
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # xi vector
        mu = o_i.mean(0)
        mu_vector[c, :] = mu

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        # Get Center Pixel
        center_pixel = coordinates.mean(0)
        # Get offset vector
        o_hat = coordinates - center_pixel

        # Regression 
        lv_vector = torch.norm(o_i - o_hat, p=2, dim=1) 
        lv_term = lv_vector - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        offset_loss += (lv_term / Nc)
        ############## Sigma ##############
        # Regression of sigma
        #sigma_i = sigma_pixels[idx_c[:, 0]]
        #sigma_k = torch.sum(sigma_i) / Nc

        #sigma_object_loss = torch.norm(sigma_i - sigma_k, p=2)
        #sigma_loss += torch.sum(sigma_object_loss, dim=0) / Nc

        ############## Offset ##############
        
        # I need to assign these values to the image vector
        probs_results[idx_c[:, 0]] = lv_vector

    '''
    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            loss_mu += ld_term
    '''
    ############## Offset Vector Loss ##############
    probs[idx_array[:, 0]] = probs_results
    # labels_binary = (labels > 0).float()
    # offset_loss = lovasz_hinge_flat(probs, labels_binary) 

    ############## Instance Vector Loss ##############
    centroid_pixels = centroid_pixels.contiguous().view(-1)
    loss_centroid = torch.sum(torch.norm(centroid_pixels - probs, p=2)) / N_pixels

    Total_Loss = offset_loss + loss_centroid# + loss_mu

    return Total_Loss




class UNet_double(nn.Module):
    def __init__(self, n_channels, n_classes_d1, n_classes_d2, num_dims=64):
        super(UNet_double, self).__init__()
        # Encoder
        self.inc = inconv(n_channels, num_dims)
        self.down1 = down(num_dims * 1, num_dims * 2)
        self.down2 = down(num_dims * 2, num_dims * 4)
        self.down3 = down(num_dims * 4, num_dims * 8)
        self.down4 = down(num_dims * 8, num_dims * 8)

        # Decoder
        self.up1 = up(num_dims * 16, num_dims * 4)
        self.up2 = up(num_dims * 8, num_dims * 2)
        self.up3 = up(num_dims * 4, num_dims * 1)
        self.up4 = up(num_dims * 2, num_dims * 1)
        self.out_d1 = outconv(num_dims, n_classes_d1)
        self.out_d2 = outconv(num_dims, n_classes_d2)


    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder1
        x_d1 = self.up1(x5, x4)
        x_d1 = self.up2(x_d1, x3)
        x_d1 = self.up3(x_d1, x2)
        x_d1 = self.up4(x_d1, x1)
        x_d1 = self.out_d1(x_d1)

        # Decoder2
        x_d2 = self.up1(x5, x4)
        x_d2 = self.up2(x_d2, x3)
        x_d2 = self.up3(x_d2, x2)
        x_d2 = self.up4(x_d2, x1)
        x_d2 = self.out_d2(x_d2)

        return [x_d1, x_d2]

    def forward_inference_offset(self, x, final_pred, eps_param=0.4, min_samples_param=10, gt=None):
        # mathilde
        device = x.device
        cube_size = x.shape[-1]

        output = self(x)

        magnitudes = output[0]
        offset_vectors = output[1]
        # tensors_io.save_volume_h5(magnitudes[0, 0, ...].cpu().numpy(), "mags", "mags")
        # exit()

        final_pred = (magnitudes > 0.8).long()
        temp = offset_vectors.float() * final_pred.float()
        temp = temp[0, ...].cpu().numpy()
        # np.save("temp_save.npy", temp)
        # exit()

        offset_vectors = offset_vectors.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

        object_indexes = (final_pred == 1).long().view(-1).nonzero()

        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(offset_vectors, 0, object_indexes.repeat(1, 3))
        a = torch.norm(object_pixels, dim=1, p=2)

        # object_pixels = object_pixels / torch.norm(object_pixels, p=2, dim=1).unsqueeze(1)

        # Get coordinates of objects
        coordinates = (final_pred[0, 0, ...] == 1).nonzero().float()

        object_pixels = coordinates - object_pixels
        # Numpy

        X = object_pixels.detach().cpu().numpy()

        # Cluster
        labels = DBSCAN(eps=eps_param, min_samples=min_samples_param).fit_predict(X)

        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        '''
        space_labels2 = torch.zeros_like(final_pred[0, 0, ...])
        space_labels2[object_pixels.long().split(1, dim=1)] = 1
        space_labels2 = space_labels2.view(cube_size, cube_size, cube_size).cpu().numpy()
        tensors_io.save_volume_h5(space_labels2, "offset_vectors", "offset_vectors")
        exit()
        '''

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(space_labels)

        return space_labels, fiber_list

    def forward_inference_debug(self, x, final_pred, eps_param=0.4, min_samples_param=10, gt=None):
        # mathilde
        device = x.device
        cube_size = x.shape[-1]

        output = self(x)

        magnitudes = output[0]
        embedding_output = output[1]

        final_pred = (magnitudes > 0.5).long()
        temp = embedding_output.float() * final_pred.float()
        temp = temp[0, ...].cpu().detach().numpy()
        # np.save("temp_save.npy", temp)
        # exit()

        embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

        object_indexes = (final_pred == 1).long().view(-1).nonzero()

        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))

        # object_pixels = object_pixels / torch.norm(object_pixels, p=2, dim=1).unsqueeze(1)

        # Get coordinates of objects
        coordinates = (final_pred[0, 0, ...] == 1).nonzero().float()

        object_pixels = coordinates - object_pixels
        # Numpy

        X = object_pixels.detach().cpu().numpy()

        # Cluster
        labels = DBSCAN(eps=1, min_samples=min_samples_param).fit_predict(X)

        space_labels = torch.zeros_like(final_pred.long().view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        space_labels2 = torch.zeros_like(gt[0, 0, ...])
        space_labels3 = torch.zeros_like(gt[0, 0, ...])
        for l in torch.unique(gt):
            if(l == 0):
                continue
            coordinates = (gt[0, 0, ...].long() == l).nonzero().float()
            object_indexes = (gt.long() == l).long().view(-1).nonzero()
            object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))
            object_pixels = coordinates - object_pixels
            space_labels2[object_pixels.long().split(1, dim=1)] = l

        for l in torch.unique(space_labels):
            if(l == 0):
                continue
            coordinates = (space_labels.long() == l).nonzero().float()
            object_indexes = (space_labels.long() == l).long().view(-1).nonzero()
            object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))
            object_pixels = coordinates - object_pixels
            space_labels3[object_pixels.long().split(1, dim=1)] = l
        
        space_labels2 = space_labels2.view(cube_size, cube_size, cube_size).cpu().numpy()
        space_labels3 = space_labels3.view(cube_size, cube_size, cube_size).cpu().numpy()
        tensors_io.save_volume_h5(space_labels2, "offset_vectors", "offset_vectors")
        tensors_io.save_volume_h5(space_labels3, "offset_vectors_inference", "offset_vectors_inference")
        tensors_io.save_volume_h5(gt[0,0,...].cpu().detach().numpy(), "offset_gt", "offset_gt")
        exit()

       

        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(space_labels)

        return space_labels, fiber_list
########################################################################################################################

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_dims=64):
        super(UNet, self).__init__()
        self.n_embeddings = n_classes
        self.inc = inconv(n_channels, num_dims)
        self.down1 = down(num_dims * 1, num_dims * 2)
        self.down2 = down(num_dims * 2, num_dims * 4)
        self.down3 = down(num_dims * 4, num_dims * 8)
        self.down4 = down(num_dims * 8, num_dims * 8)
        self.up1 = up(num_dims * 16, num_dims * 4)
        self.up2 = up(num_dims * 8, num_dims * 2)
        self.up3 = up(num_dims * 4, num_dims * 1)
        self.up4 = up(num_dims * 2, num_dims * 1)
        self.outc = outconv(num_dims, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def forward_inference_offset(self, x, final_pred, eps_param=0.4, min_samples_param=10, gt=None):
        # mathilde
        device = x.device
        cube_size = final_pred.shape[-1]

        embedding_output = self(x)

        temp = embedding_output.float() * final_pred.float()
        temp = temp[0, ...].cpu().numpy()
        # np.save("temp_save.npy", temp)
        # exit()

        embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

        object_indexes = (final_pred == 1).long().view(-1).nonzero()

        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))
        a = torch.norm(object_pixels, dim=1, p=2)

        # object_pixels = object_pixels / torch.norm(object_pixels, p=2, dim=1).unsqueeze(1)

        # Get coordinates of objects
        coordinates = (final_pred[0, 0, ...] == 1).nonzero().float()

        object_pixels = coordinates - object_pixels
        # Numpy

        X = object_pixels.detach().cpu().numpy()

        # Cluster
        labels = DBSCAN(eps=eps_param, min_samples=min_samples_param).fit_predict(X)

        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        '''
        space_labels2 = torch.zeros_like(final_pred[0, 0, ...])
        space_labels2[object_pixels.long().split(1, dim=1)] = 1
        space_labels2 = space_labels2.view(cube_size, cube_size, cube_size).cpu().numpy()
        tensors_io.save_volume_h5(space_labels2, "offset_vectors", "offset_vectors")
        exit()
        '''

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(space_labels)

        return space_labels, fiber_list

    def forward_inference_debug(self, x, final_pred, eps_param=0.4, min_samples_param=10, gt=None):
        # mathilde
        device = x.device
        cube_size = final_pred.shape[-1]

        embedding_output = self(x)

        temp = embedding_output.float() * final_pred.float()
        temp = temp[0, ...].cpu().detach().numpy()
        # np.save("temp_save.npy", temp)
        # exit()

        embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

        object_indexes = (final_pred == 1).long().view(-1).nonzero()

        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))

        # object_pixels = object_pixels / torch.norm(object_pixels, p=2, dim=1).unsqueeze(1)

        # Get coordinates of objects
        coordinates = (final_pred[0, 0, ...] == 1).nonzero().float()

        object_pixels = coordinates - object_pixels
        # Numpy

        X = object_pixels.detach().cpu().numpy()

        # Cluster
        labels = DBSCAN(eps=1, min_samples=min_samples_param).fit_predict(X)

        space_labels = torch.zeros_like(final_pred.long().view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        space_labels2 = torch.zeros_like(gt[0, 0, ...])
        space_labels3 = torch.zeros_like(gt[0, 0, ...])
        for l in torch.unique(gt):
            if(l == 0):
                continue
            coordinates = (gt[0, 0, ...].long() == l).nonzero().float()
            object_indexes = (gt.long() == l).long().view(-1).nonzero()
            object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))
            object_pixels = coordinates - object_pixels
            space_labels2[object_pixels.long().split(1, dim=1)] = l

        for l in torch.unique(space_labels):
            if(l == 0):
                continue
            coordinates = (space_labels.long() == l).nonzero().float()
            object_indexes = (space_labels.long() == l).long().view(-1).nonzero()
            object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, 3))
            object_pixels = coordinates - object_pixels
            space_labels3[object_pixels.long().split(1, dim=1)] = l
        
        space_labels2 = space_labels2.view(cube_size, cube_size, cube_size).cpu().numpy()
        space_labels3 = space_labels3.view(cube_size, cube_size, cube_size).cpu().numpy()
        tensors_io.save_volume_h5(space_labels2, "offset_vectors", "offset_vectors")
        tensors_io.save_volume_h5(space_labels3, "offset_vectors_inference", "offset_vectors_inference")
        tensors_io.save_volume_h5(gt[0,0,...].cpu().detach().numpy(), "offset_gt", "offset_gt")
        exit()

       

        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(space_labels)

        return space_labels, fiber_list

    def forward_inference_direction(self, x, final_pred):
        masks_pred_temp = self(x)
        final_seg = final_pred.repeat((1, 3, 1, 1, 1))
        masks_pred_temp = masks_pred_temp * final_seg.float()
        return masks_pred_temp

    def forward_inference_fast(self, x, final_pred, eps_param=0.4, min_samples_param=10, p=-1):
        device = x.device
        cube_size = final_pred.shape[-1]

        embedding_output = self(x)

        embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_embeddings)

        object_indexes = (final_pred == 1).long().view(-1).nonzero()

        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, self.n_embeddings))

        # Numpy
        X = object_pixels.detach().cpu().numpy()

        # Cluster
        labels = DBSCAN(eps=eps_param, min_samples=min_samples_param).fit_predict(X)
        
        # Pytorch
        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        space_labels = refine_connected(space_labels)
        # space_labels = refine_watershed(space_labels)

        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(space_labels)
        
        #end_points_image = torch.zeros_like(space_labels)
        '''
        if(p == 6):
            tensors_io.save_volume_h5(space_labels.cpu().numpy().astype(np.int64), directory='./h5_files', name='pre_merged', dataset_name='pre_merged')
        '''

        # X = refine_watershed_end_points(end_points.cpu().numpy())
        if(len(end_points) > 10):
            merge_inner_fibers(end_points, fiber_list, fiber_ids, space_labels)

        return space_labels, fiber_list

    def forward_inference(self, x, final_pred, num_fibers=0, eps_param=0.4, min_samples_param=10, device=None):
        GPU_YES = torch.cuda.is_available()
        
        if(device is None):
            device = torch.device("cuda:0" if GPU_YES else "cpu")
        cube_size = final_pred.shape[-1]

        embedding_output = self(x)

        embedding_output = embedding_output.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_embeddings)

        object_indexes = (final_pred > 0).long().view(-1).nonzero()

        #np.save("final_pred", final_pred)
        #exit()
        if(len(object_indexes) == 0):
            return(None)
        object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, self.n_embeddings))

        # Numpy
        X = object_pixels.detach().cpu().numpy()

        labels = DBSCAN(eps=eps_param, min_samples=min_samples_param).fit_predict(X)
        #labels = MeanShift(bandwidth=0.4).fit_predict(X)
        space_labels = torch.zeros_like(final_pred.view(-1))
        space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

        space_labels = space_labels.view(cube_size, cube_size, cube_size)

        space_labels = refine_watershed(space_labels, final_pred)
        space_labels = space_labels.to(torch.device("cpu")).numpy()

        pre_cleaned_list = fit_all_fibers_parallel(space_labels, np.array([0, 0, 0]))

        (fiber_list, volume_to_correct) = post_processing(pre_cleaned_list, space_labels)


        # second pass
        # volume_to_correct = torch.from_numpy(volume_to_correct).cuda().unsqueeze(0)


        '''
        object_indexes = (volume_to_correct > 0).long().view(-1).nonzero()
        if(len(object_indexes) >  0):
            object_pixels = torch.gather(embedding_output, 0, object_indexes.repeat(1, self.n_embeddings))
            # Numpy
            X = object_pixels.detach().cpu().numpy()
            labels = DBSCAN(eps=eps_param - 0.05, min_samples=5).fit_predict(X)
      
            space_labels2 = torch.zeros_like(final_pred.view(-1))
            space_labels2[object_indexes] = torch.from_numpy(labels).unsqueeze(1).to(device) + 2

            space_labels2 = space_labels2.view(cube_size, cube_size, cube_size)

            space_labels2 = refine_watershed(space_labels2, volume_to_correct)
            space_labels2 = space_labels2.to(torch.device("cpu")).numpy()

            pre_cleaned_list = fit_all_fibers_parallel(space_labels2, np.array([0, 0, 0]))
            
            (fiber_list2, volume_to_correct) = post_processing(pre_cleaned_list, space_labels2)
            id_offset = space_labels.max()
            space_labels2[np.where(space_labels2 > 0)] += id_offset

            space_labels[np.where(space_labels == 0)] += space_labels2[np.where(space_labels == 0)]
            for el in fiber_list2:
                el[0] += id_offset
                fiber_list.append(el)
        '''

        return (space_labels, fiber_list, volume_to_correct.astype(np.long))

def embedded_geometric_loss(outputs, labels, mini_v):
    delta_v = 0.2
    delta_d = 5

    alpha = 2
    beta = 2
    gamma = 0.0000001
    GPU_YES = torch.cuda.is_available()
    device = torch.device("cuda" if GPU_YES else "cpu")

    N_embedded = outputs.shape[1]
    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)

    # Find mu vector
    Lv = 0  # Lv_same_object_proximity
    Lr = 0  # Ld_minimal_distance_clusters
    Ld = 0  # Lr_cluster_center_regularizer
    for c in range(N_objects):
        fiber_id = labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()
        coordinates = (mini_v == fiber_id).nonzero().float()
        # xi vector
        x_i = torch.gather(object_pixels, 0, idx_c.repeat(1, N_embedded))
        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu

        # get Lv
        Nc = len(idx_c)
        weights = r_individual(coordinates)
        if (1 in torch.isnan(weights)):
            weights = torch.ones(Nc, device=coordinates.device)
        else:
            weights = 1 / (1 + torch.exp(- weights))

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2

        lv_term = lv_term * weights
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = alpha * Lv + beta * Ld + gamma * Lr
    return loss


def embedded_geometric_loss(outputs, labels, mini_v):
    delta_v = 0.2
    delta_d = 5

    alpha = 2
    beta = 2
    gamma = 0.0000001
    GPU_YES = torch.cuda.is_available()
    device = torch.device("cuda" if GPU_YES else "cpu")

    N_embedded = outputs.shape[1]
    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)

    # Find mu vector
    Lv = 0  # Lv_same_object_proximity
    Lr = 0  # Ld_minimal_distance_clusters
    Ld = 0  # Lr_cluster_center_regularizer
    for c in range(N_objects):
        fiber_id = labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()
        coordinates = (mini_v == fiber_id).nonzero().float()
        # xi vector
        x_i = torch.gather(object_pixels, 0, idx_c.repeat(1, N_embedded))
        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu

        # get Lv
        Nc = len(idx_c)
        weights = r_individual(coordinates)
        if (1 in torch.isnan(weights)):
            weights = torch.ones(Nc, device=coordinates.device)
        else:
            weights = 1 / (1 + torch.exp(- weights))

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2

        lv_term = lv_term * weights
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = alpha * Lv + beta * Ld + gamma * Lr
    return loss
def embedded_loss(outputs, labels):
    delta_v = 0.2
    delta_d = 5

    alpha = 2
    beta = 2
    gamma = 0.0000001
    GPU_YES = torch.cuda.is_available()
    device = torch.device("cuda:0" if GPU_YES else "cpu")

    N_embedded = outputs.shape[1]
    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)

    # Find mu vector
    Lv = 0  # Lv_same_object_proximity
    Lr = 0  # Ld_minimal_distance_clusters
    Ld = 0  # Lr_cluster_center_regularizer
    for c in range(N_objects):
        fiber_id = labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()

        # xi vector
        x_i = torch.gather(object_pixels, 0, idx_c.repeat(1, N_embedded))
        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu

        # get Lv
        Nc = len(idx_c)

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = alpha * Lv + beta * Ld + gamma * Lr
    return loss



def embedded_directional_loss(network_outputs, labels):
    delta_v = 0.2
    delta_d = 5

    alpha = 2
    beta = 2
    gamma = 0.0000001


    GPU_YES = torch.cuda.is_available()
    device = torch.device("cuda:0" if GPU_YES else "cpu")

    outputs = network_outputs[0]
    directions = network_outputs[1]

    N_embedded = outputs.shape[1]
    N_directions = directions.shape[1]
    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_embedded))
    direction_pixels = torch.gather(outputs, 0, idx_array.repeat(1, N_directions))

    labels = torch.unique(labeled_pixels, sorted=True)
    N_objects = len(labels)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)

    # Find mu vector
    Lv = 0  # Lv_same_object_proximity
    Lr = 0  # Ld_minimal_distance_clusters
    Ld = 0  # Lr_cluster_center_regularizer
    for c in range(N_objects):
        fiber_id = labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()

        # xi vector
        x_i = torch.gather(object_pixels, 0, idx_c.repeat(1, N_embedded))
        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu

        # get Lv
        Nc = len(idx_c)

        lv_term = torch.norm(mu_vector[c, :] - x_i, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = lv_term**2
        lv_term = torch.sum(lv_term, dim=0)
        Lv += (lv_term / Nc)
        Lr += torch.norm(mu, 2)

    for ca in range(N_objects):
        for cb in range(ca + 1, N_objects):
            ld_term = delta_d - torch.norm(mu_vector[ca, :] - mu_vector[cb, :], 2)
            ld_term = torch.clamp(ld_term, 0, 10000000)
            ld_term = ld_term ** 2
            Ld += ld_term

    N_objects = float(N_objects)
    Lv /= N_objects
    Lr /= N_objects
    Ld /= (N_objects * (N_objects - 1))

    loss = alpha * Lv + beta * Ld + gamma * Lr
    return loss

def distance_loss(network_outputs, labels):
    labels = labels.contiguous().view(-1)
    network_outputs = network_outputs.contiguous().view(-1)

    idx_array = (labels).nonzero().split(1, dim=1)



    object_pixels = network_outputs[idx_array].double()
    label_pixels = labels[idx_array].double()
    # idx at fibers
    # Get only the non-zero indexes
    loss = torch.norm(label_pixels - object_pixels, p=2)
    return loss

def direction_loss(network_outputs, labels, device=None):
    idx_array = (labels[:, 0] + labels[:, 1] + labels[:, 2]).nonzero()

    object_pixels = torch.gather(network_outputs, 0, idx_array.repeat(1,  3))
    label_pixels = torch.gather(labels, 0, idx_array.repeat(1, 3))
    # idx at fibers
    # Get only the non-zero indexes
    loss = torch.norm(label_pixels - object_pixels, p=1)
    return loss



def refine_connected(labels):
    device = labels.device
    labels = labels.cpu().numpy()
    labels_t = np.zeros(labels.shape)
    num_labels = labels.max().astype(np.int)
    counter = 2

    for c in range(2, num_labels + 1):
        im = labels == c
        temp_labels, temp_nums = measure.label(im, return_num=True)
        for points in range(1, temp_nums + 1):
            idx_c = np.where(temp_labels == points)
            if(len(idx_c[0]) < 30):
                labels_t[idx_c] = 1
            else:
                labels_t[idx_c] = counter
                counter = counter + 1
    labels_t[np.where(labels == 1)] = 1
    labels = torch.from_numpy(labels_t).long().to(device)
    return (labels)

def refine_watershed_end_points(labels):
    num_labels = labels.max().astype(np.int)
    end_points = []
    label_points = []
    for c in range(2, num_labels + 1):
        im = labels == c
        temp_labels, temp_nums = measure.label(im, return_num=True)
        for end_point_idx in range(1, temp_nums + 1):
            idx_c = np.where(temp_labels == end_point_idx)
            mean =[X.mean() for X in idx_c]
            end_points.append(mean)
            label_points.append(c)
    return (end_points, label_points)


def refine_watershed(labels, segmentation=None):
    device = labels.device
    labels = labels.cpu().numpy()

    markers = np.copy(labels)
    markers[np.where(labels == 1)] = 0

    energy = np.zeros(labels.shape)
    energy[np.where(labels==1)] = 1
    distance = ndi.distance_transform_edt(energy)
    distance[np.where(labels > 1)] = 1

    mask = np.zeros(labels.shape)
    mask[np.where(labels > 0)] = 1
    labels = watershed(-distance, markers, mask=mask)

    labels= torch.from_numpy(labels).long().to(device)

    return labels


def save_data(embeddings, labels, iteration=None, detected_labels=None):
    N_embedded = embeddings.shape[1]
    # num_dims = N_embedded
    # num_display_dims = 2
    # tsne_lr = 20.0

    # Get only the non-zero indexes
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(embeddings, 0, idx_array.repeat(1, N_embedded))

    from sklearn.manifold import TSNE
    # from tsne import tsne
    X = object_pixels.cpu().detach().numpy()
    Y = TSNE(n_components=2, perplexity=40, learning_rate=50).fit_transform(X)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(Y[:, 0], Y[:, 1], 5, labeled_pixels, cmap='tab20b')
    if iteration is None:
        iteration = 0
    plt.savefig("low_dim_embeeding/embedded_%d.png" % iteration)
    plt.close(fig)


    if detected_labels is not None:
        detected_labels_pixels = detected_labels[idx_tuple].squeeze(1)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        pylab.scatter(Y[:, 0], Y[:, 1], 5, detected_labels_pixels, cmap='tab20b')
        pylab.savefig("low_dim_embeeding/embedded_%d.png" % (iteration + 1))
        pylab.close(fig)

def get_single_fiber_property(space_labels, fiber_id):
    idx = (space_labels == fiber_id).nonzero().float()
     
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
          
    # Find farthest point from end point 1
    rs2 = torch.norm(idx - end_point0, p=2, dim=1)
    # Find farthest point from end point 1
    end_point2_idx = (rs2 > rs2.max() - 3).nonzero()
    end_point2_idx = end_point2_idx[:, 0]
    end_point2 = torch.tensor([idx_split[i][end_point2_idx][:, 0].mean() for i in range(3)])

    c_np = center.cpu().numpy()

    length = torch.norm(end_point1 - end_point2, p=2).cpu().item()
    direction = (end_point1 - end_point2)
    direction = direction / torch.norm(direction, p=2)

    R = 1.5 # rr[1]

    direction = direction.cpu().numpy()
    return  [fiber_id, c_np[0], c_np[1], c_np[2], R, length, direction[0], direction[1], direction[2]]

def get_fiber_properties(space_labels, large_volume=False):
    end_points = []
    fiber_ids = []
    centers = {}
    fiber_list = {}
    # id center1, center2, center3, L, R, Ty, Tz, error
    for fiber_id in torch.unique(space_labels):
        if(fiber_id == 0 or fiber_id == 1):
            continue
        idx = (space_labels == fiber_id).nonzero().float()
        if(large_volume is True):
            if(len(idx) < 20):
                space_labels[idx.long().split(1, dim=1)] = 0
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


        '''
        close_to_end_point1 = (rs2 < 3).nonzero().long()
        close_to_end_point1 = close_to_end_point1[:, 0]
        end_points_image[idx_split[0][close_to_end_point1].long(), idx_split[1][close_to_end_point1].long(), idx_split[2][close_to_end_point1].long()] = fiber_id
        close_to_end_point2 = (rs2 > rs2.max() - 3).nonzero().long()
        close_to_end_point2 = close_to_end_point2[:, 0]
        end_points_image[idx_split[0][close_to_end_point2].long(), idx_split[1][close_to_end_point2].long(), idx_split[2][close_to_end_point2].long()] = fiber_id
        '''
        end_points.append(end_point1.cpu().numpy())
        end_points.append(end_point2.cpu().numpy())

        fiber_ids.append(fiber_id.cpu().item())
        fiber_ids.append(fiber_id.cpu().item())

        c_np = center.cpu().numpy()
        centers[fiber_id.cpu().item()] = c_np

        length = torch.norm(end_point1 - end_point2, p=2).cpu().item()
        direction = (end_point1 - end_point2)
        direction = direction / torch.norm(direction, p=2)

        '''
        if(math.isnan(direction)):
            idx = (space_labels == fiber_id).nonzero().split(1, dim=1)
            space_labels[idx] = 1
        '''
       #  rr = fit_t.r2(direction.unsqueeze(1), idx, center.unsqueeze(1))
        R = 1.5 # rr[1]
        # G = fit_t.G(direction.unsqueeze(1), idx)

        direction = direction.cpu().numpy()
        fiber_list[fiber_id.cpu().item()] = [fiber_id.cpu().item(), c_np[0], c_np[1], c_np[2], R, length, direction[0], direction[1], direction[2]]

    return centers, fiber_ids, end_points, fiber_list


def evaluate_iou(Vf, Vgt):
    Vf[np.where(Vf == 1)] = 0
    # Vf[np.where(Vgt == 0)] = 0
    labels_gt = np.unique(Vgt)
    num_fibers = len(labels_gt) - 1

    labels_f = np.unique(Vf)
    num_fibers_f = len(labels_f) - 1

    set_f = set(labels_f)
    set_gt = set(labels_gt)

    set_f.remove(0)
    set_gt.remove(0)

    Vf = Vf[0:150, 0:150, 0:150]
    print("Num Fibers Gt: {}".format(num_fibers))
    print("Labels Vf:{}".format(num_fibers_f))

    fibers_corrected_detected = 0
    fibers_splitted_but_detected = 0

    fibers_in_v_detected_double = 0
    flag_match_detected = 0
    for Lgt in set_gt:

        Vgt_temp = np.zeros(Vgt.shape)
        idxs_gt = np.where(Vgt == Lgt)
        Vgt_temp[idxs_gt] = 1

        labels_in_V = set(np.unique(Vf[idxs_gt]))
        labels_in_V = labels_in_V.intersection(set_f)

        IOU = 0.0
        total_intersection = 0
        set_broken_fibers = set()
        for Lf in labels_in_V:
            Vf_temp = np.zeros(Vgt.shape)
            idxs_f = np.where(Vf == Lf)
            Vf_temp[idxs_f] = 1

            #num_detected_gt_fibers = len(np.unique(Vgt[idxs_f]))
            #if(num_detected_gt_fibers > 2):
            #    fibers_in_v_detected_double += 1

            intersection = np.logical_and(Vgt_temp, Vf_temp).sum().astype(np.float)
            union = np.logical_or(Vgt_temp, Vf_temp).sum().astype(np.float)
            area = (Vgt_temp).sum().astype(np.float)

            ind_IOU = intersection / union

            # print(area)
            if(ind_IOU > 0.5):
                flag_match_detected = 1
                fibers_corrected_detected += 1
                set_f.remove(Lf)
                IOU = ind_IOU
                # print("IOU", IOU)
                break
            else:
                total_intersection += intersection
                set_broken_fibers.add(Lf)

        if(not flag_match_detected):
            print("Cheking again")
            IOU = total_intersection / union
            if(IOU > 0.5):
                fibers_splitted_but_detected += 1
                for item in set_broken_fibers:
                    set_f.remove(item)
    print("")
    print("Total Fibers: {}, Fibers Detected and Splitted {}, Fibers Correctly Detected {}".format(num_fibers, fibers_splitted_but_detected, fibers_corrected_detected))
    print("Percent of Fibers Correctly Detected: {}".format(float(fibers_corrected_detected) / float(num_fibers)))
    print("Percent of Fibers Detected ans splitted: {}".format(float(fibers_splitted_but_detected) / float(num_fibers)))

    print("")
    print("Fibers Missed: {}".format(num_fibers - fibers_corrected_detected - fibers_splitted_but_detected))
    print("Fibers in V that detected double {}".format(fibers_in_v_detected_double))


def merge_inner_fibers(end_points, fiber_list, fiber_ids, space_labels, debug=0, radius=10, angle_threshold=10):
    neigh = NearestNeighbors(n_neighbors=4, radius=2).fit(end_points)
    A = neigh.kneighbors_graph(end_points, mode='distance')        # A = A.toarray()
    sure_neighbors = []
    lost_ids_dict = {}
    mini_fibers = {}
    A = find_sparse(A)
    fiber_merged_dict = defaultdict(list)



    if(debug):
        f = open("instances/debug/post_merged/merging_fibers.txt", "w")

    for i in range(len(A[0])):
        i_entry = A[0][i]
        j_entry = A[1][i]
        distance = A[2][i]
        nan_flag = 0
        nan_flag_a = 0
        nan_flag_b = 0

        if(i_entry == j_entry):
            continue
        if(fiber_ids[i_entry] == fiber_ids[j_entry] or distance > radius):
            continue

        if( ((fiber_ids[j_entry], fiber_ids[i_entry]) in sure_neighbors) or ((fiber_ids[i_entry], fiber_ids[j_entry]) in sure_neighbors)):
            continue

        fiber_a = fiber_list[fiber_ids[i_entry]]
        fiber_b = fiber_list[fiber_ids[j_entry]]

        dir_a = np.array([fiber_a[6], fiber_a[7], fiber_a[8]])
        dir_b = np.array([fiber_b[6], fiber_b[7], fiber_b[8]])
        

        center_a = np.array([fiber_a[1], fiber_a[2], fiber_a[3]])
        center_b = np.array([fiber_b[1], fiber_b[2], fiber_b[3]])
      
        # Vector between centers 
        vector_between_centers = center_a - center_b
        vector_between_centers = vector_between_centers / np.sqrt(np.dot(vector_between_centers, vector_between_centers))
 
        angle_between = np.arccos(np.abs(np.dot(dir_a, dir_b))) * 180 / np.pi
        if(math.isnan(angle_between)):
            angle_between = 0
            nan_flag = 1

        angle_between_a_center_v = np.arccos(np.abs(np.dot(dir_a, vector_between_centers))) * 180 / np.pi
        if(math.isnan(angle_between_a_center_v)):
            angle_between_a_center_v = 0
            nan_flag_a = 1

        angle_between_b_center_v = np.arccos(np.abs(np.dot(dir_b, vector_between_centers))) * 180 / np.pi

        if(math.isnan(angle_between_b_center_v)):
            angle_between_b_center_v = 0
            nan_flag_b = 1

        angle_total = max(angle_between_a_center_v, angle_between_b_center_v)


        if(debug):

            f.write("Considering {} {}\n".format(fiber_ids[i_entry], fiber_ids[j_entry]))

            Txya = np.arctan2(dir_a[1], dir_a[0]) * 180 / np.pi
            if(Txya < 0):
                Txya = 180 + Txya
            Tza = np.arccos(np.dot(dir_a, np.array([0, 0, 1])) / np.linalg.norm(dir_a, 2)) * 180 / np.pi

            Txyb = np.arctan2(dir_b[1], dir_b[0]) * 180 / np.pi
            if(Txyb < 0):
                Txyb = 180 + Txyb
            Tzb = np.arccos(np.dot(dir_b, np.array([0, 0, 1])) / np.linalg.norm(dir_b, 2)) * 180 / np.pi


            Txyc = np.arctan2(vector_between_centers[1], vector_between_centers[0]) * 180 / np.pi
            if(Txyc < 0):
                Txyc = 180 + Txyc
            Tzc = np.arccos(np.dot(vector_between_centers, np.array([0, 0, 1])) / np.linalg.norm(vector_between_centers, 2)) * 180 / np.pi

            f.write("angle_between {}\n".format(angle_between))
            f.write("angle_between_a_center_v {}\n".format(angle_between_a_center_v))
            f.write("angle_between_b_center_v {}\n".format(angle_between_b_center_v))
            f.write("Center a: {}, {}, {}\n".format(center_a[0], center_a[1], center_a[2]))
            f.write("Center b: {}, {}, {}\n".format(center_b[0], center_b[1], center_b[2]))

            f.write("Txy_a, Tz_a: {}, {}\n".format(Txya, Tza))
            f.write("Txy_b, Tz_b: {}, {}\n".format(Txyb, Tzb))
            f.write("Txy_c, Tz_c: {}, {}\n".format(Txyc, Tzc))


        if(angle_total < 5 and angle_between < 10 and nan_flag == 0):
            if((fiber_ids[j_entry], fiber_ids[i_entry]) not in sure_neighbors and (fiber_ids[i_entry], fiber_ids[j_entry]) not in sure_neighbors):
                sure_neighbors.append((fiber_ids[j_entry], fiber_ids[i_entry]))
                fiber_merged_dict[fiber_ids[j_entry]].append(fiber_ids[i_entry])
        
        # fibers that are too small to get an angle:
        elif(angle_total < angle_threshold and nan_flag == 1):
            # Consider fiber a is a very very small
            if(nan_flag_a == 1 and nan_flag_b == 0):
                # If fiber a has not been seen yet
                if(fiber_ids[i_entry] not in mini_fibers.keys()):
                    mini_fibers[fiber_ids[i_entry]] = (fiber_ids[j_entry], distance)
                else:
                # Check if the new candidate is better for fiber_a
                    if distance < mini_fibers[fiber_ids[i_entry]][1]:
                        mini_fibers[fiber_ids[i_entry]] = (fiber_ids[j_entry], distance)

            # Consider fiber b is a very very small
            elif(nan_flag_b == 1 and nan_flag_a == 0):
                # If fiber b has not been seen yet
                if(fiber_ids[j_entry] not in mini_fibers.keys()):
                    mini_fibers[fiber_ids[j_entry]] = (fiber_ids[i_entry], distance)
                else:
                # Check if the new candidate is better for fiber_b
                    if distance < mini_fibers[fiber_ids[j_entry]][1]:
                        mini_fibers[fiber_ids[j_entry]] = (fiber_ids[i_entry], distance)

            #if both fibers are very very small
            else:
                if(fiber_ids[i_entry] not in mini_fibers.keys()):
                    mini_fibers[fiber_ids[i_entry]] = (fiber_ids[j_entry], 10000)
                
                if(fiber_ids[j_entry] not in mini_fibers.keys()):
                    mini_fibers[fiber_ids[j_entry]] = (fiber_ids[i_entry], 10000)


    for (fa, fb) in sure_neighbors:
        if(debug):
            f.write("Merging {} {}\n".format(fa, fb))

        while(fb in lost_ids_dict.keys()):
            #print(lost_ids_dict[fb])
            fb = lost_ids_dict[fb]
            #print("Saved one lost id")

        while(fa in lost_ids_dict.keys()):
            # print(lost_ids_dict[fa])
            fa = lost_ids_dict[fa]
            # print("Saved another lost id")

        if(fa != fb):
            lost_ids_dict[fa] = fb
        else:
            continue

        idx1 = (space_labels == fa).nonzero()
        if(len(idx1) == 0):
            continue
        idx1 = idx1.split(1, dim=1)
        space_labels[idx1] = fb

        new_entry = get_single_fiber_property(space_labels, fb)
        fiber_list[fb] = new_entry
        del fiber_list[fa]



    # Take Care of Very Small Fibers
    for fa in mini_fibers.keys():
        '''
        fb = mini_fibers[fa][0]
        dist = mini_fibers[fa][1]
        while(fb in lost_ids_dict.keys()):
            fb = lost_ids_dict[fb]
        
        if(debug):
            f.write("Merging Very Small {} {} {}\n".format(fa, fb, dist ))

        if(fa != fb):
            lost_ids_dict[fa] = fb
        '''
        idx1 = (space_labels == fa).nonzero()
        if(len(idx1) == 0):
            continue
        idx1 = idx1.split(1, dim=1)
        space_labels[idx1] = 0

    if(debug):
        f.close()
    return len(lost_ids_dict)


def merge_outer_fibers(end_points, centers, fiber_ids, space_labels, debug=0, radius=10):
    neigh = NearestNeighbors(n_neighbors=8, radius=radius).fit(end_points)
    A = neigh.kneighbors_graph(end_points, mode='distance')        # A = A.toarray()
    possible_neighbors = set()
    sure_neighbors = []
    lost_ids_dict = {}

    A = find_sparse(A)
    fiber_merged_dict = defaultdict(list)

    device = space_labels.device
    space_labels = space_labels.cpu().numpy()

    for i in range(len(A[0])):
        i_entry = A[0][i]
        j_entry = A[1][i]

        if(i_entry == j_entry):
            continue
        if(fiber_ids[i_entry] == fiber_ids[j_entry] or A[2][i] > radius):
            continue

        # [L, C_fit[0], C_fit[1], C_fit[2], r_fit, h_fit, Txy, Tz, fit_err]
        # properties1 = guess_cylinder_parameters_merged(fiber_ids[i_entry], -1, space_labels)
        #properties2 = guess_cylinder_parameters_merged(fiber_ids[j_entry], -1, space_labels)

        properties_together = guess_cylinder_parameters_merged(fiber_ids[i_entry], fiber_ids[j_entry], space_labels)
        err = properties_together[-1]
        if(err < 100):
            if((fiber_ids[j_entry], fiber_ids[i_entry]) not in possible_neighbors and (fiber_ids[i_entry], fiber_ids[j_entry]) not in possible_neighbors):
                sure_neighbors.append((fiber_ids[j_entry], fiber_ids[i_entry]))
                fiber_merged_dict[fiber_ids[j_entry]].append(fiber_ids[i_entry])

    space_labels = torch.from_numpy(space_labels).to(device)

    for (fa, fb) in sure_neighbors:
        if(0):
            print("Merging {} {}".format(fa, fb))

        while(fb in lost_ids_dict.keys()):
            #print(lost_ids_dict[fb])
            fb = lost_ids_dict[fb]
            #print("Saved one lost id")

        while(fa in lost_ids_dict.keys()):
            # print(lost_ids_dict[fa])
            fa = lost_ids_dict[fa]
            # print("Saved another lost id")

        if(fa != fb):
            lost_ids_dict[fa] = fb

        idx1 = (space_labels == fa).nonzero()
        if(len(idx1) == 0):
            continue
        idx1 = idx1.split(1, dim=1)
        space_labels[idx1] = fb

    return len(sure_neighbors)

def merge_big_vol():
    Window_Size = 50
    for i in range(0, Vol.shape[-1], Window_Size):
        temp_vol = np.copy(Vol[:, :, i: i + Window_Size])
        # tensors_io.save_volume_h5(Vol, "pre_merged", "pre_merged")
        temp_vol = torch.from_numpy(temp_vol)
        print("Finding Pairs")
        centers, fiber_ids, end_points, fiber_list = get_fiber_properties(temp_vol)

        print("Merging Neighbors")
        merge_inner_fibers(end_points, centers, fiber_ids, temp_vol)
        Vol[:, :, i: i + Window_Size] = torch.from_numpy(np.copy(temp_vol))

    tensors_io.save_volume_h5(Vol.numpy(), "merged_big_vol", "merged_big_vol")


# Regression to instance center
# Mathilde
def embedded_geometric_loss_coords(outputs, labels):
    delta_v = 2
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 3))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Center Pixel
        center_pixel = coordinates.mean(0)

        # Get offset vector
        o_hat = coordinates - center_pixel
        
        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(o_i - o_hat, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss


def embedded_geometric_loss_coords22(outputs, labels):
    delta_v = 2
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    mu_vector = torch.zeros(N_objects, N_embedded).to(device)
    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 3))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # xi vector
        mu = o_i.mean(0)
        mu_vector[c, :] = mu

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Center Pixel
        center_pixel = coordinates.mean(0)

        # Get offset vector
        o_hat = coordinates - mu
        
        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(o_i - o_hat, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss

def embedded_geometric_loss_r(outputs, labels):
    delta_v = 2
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    centers, fiber_ids, end_points, fiber_list = get_fiber_properties(labels3D)
    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 3))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Center Pixel
        center_pixel = coordinates.mean(0)

        # Get offset vector
        o_hat = coordinates - center_pixel

        if(fiber_id == 1):
            continue
        properties = fiber_list[fiber_id.cpu().item()]
        W = torch.tensor(np.array([properties[6], properties[7], properties[8]]).astype(np.float)).to(device)
        W = W.unsqueeze(1).cpu().numpy()
        P = np.identity(3) - np.dot(np.reshape(W, (3, 1)), np.reshape(W, (1, 3)))
        
        P = torch.from_numpy(P).float().to(device)
        o_hat = torch.mm(P, o_hat.t())
        o_hat = o_hat.t()

        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(o_i - o_hat, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss

def embedded_geometric_loss_r(outputs, labels):
    delta_v = 2
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    centers, fiber_ids, end_points, fiber_list = get_fiber_properties(labels3D)
    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 3))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        o_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 3))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Center Pixel
        center_pixel = coordinates.mean(0)

        # Get offset vector
        o_hat = coordinates - center_pixel

        if(fiber_id == 1):
            continue
        properties = fiber_list[fiber_id.cpu().item()]
        W = torch.tensor(np.array([properties[6], properties[7], properties[8]]).astype(np.float)).to(device)
        W = W.unsqueeze(1).cpu().numpy()
        P = np.identity(3) - np.dot(np.reshape(W, (3, 1)), np.reshape(W, (1, 3)))
        
        P = torch.from_numpy(P).float().to(device)
        o_hat = torch.mm(P, o_hat.t())
        o_hat = o_hat.t()

        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(o_i - o_hat, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss


def projection_matrix(w):
    I_m = torch.eye(3)
    device = w.device
    I_m = I_m.to(device)
    mult = torch.mm(w, w.t())
    return I_m - mult

# Regression to instance center
def embedded_geometric_loss_radious(outputs, labels):
    delta_v = 0
    device = outputs.device
    labels_ids = torch.unique(labels, sorted=True)
    labels_ids = labels_ids[1:]
    N_objects = len(labels_ids)

    # copy 3D
    labels3D = labels[0, 0, ...].clone()

    # Flatten Everything adn make outputs 3 dimensions at the end
    labels = labels.contiguous().view(-1)
    outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 1)

    # idx at fibers
    idx_array = labels.nonzero()
    idx_tuple = idx_array.split(1, dim=1)

    # Get only the non-zero indexes
    labeled_pixels = labels[idx_tuple].squeeze(1)
    object_pixels = torch.gather(outputs, 0, idx_array.repeat(1, 1))
    Loss = 0
    for c in range(N_objects):
        fiber_id = labels_ids[c]

        # Get sub indexes at fiber_id
        idx_c = (labeled_pixels == fiber_id).nonzero()
        # Get output vectors at those dimensions
        r_i = torch.gather(object_pixels, 0, idx_c.repeat(1, 1))

        # Get coordinates of objects
        coordinates = (labels3D == fiber_id).nonzero().float()
        
        # Get Radious
        if(len(coordinates) < 30):
            continue
        radii = r_individual(coordinates.clone())
        raddii = radii.detach()
        
        Nc = len(idx_c)

        # Regression 
        lv_term = torch.norm(r_i - radii, p=2, dim=1) - delta_v
        lv_term = torch.clamp(lv_term, 0, 10000000)
        lv_term = torch.sum(lv_term, dim=0)
        Loss += (lv_term / Nc)

    return Loss
if __name__ == '__main__':
    Vol = tensors_io.read_volume_h5('final_fibers_single','final_fibers_single','../h5_files')
    Vol = torch.from_numpy(Vol).unsqueeze(0)
    Vol = Vol.unsqueeze(0)

    [_, _, rows, cols, slices] = Vol.shape
    embedded_geometric_loss_coords(torch.zeros(1, 3, rows, cols, slices), Vol)
    '''
    import tensors_io
    print("Reading Volume")
    Vol = tensors_io.read_volume_h5('final_fibers_single','final_fibers_single','../h5_files')
    Vol = Vol.astype(np.int32)
    Vol = Vol[0:100, 0:100, 0:100]
    tensors_io.save_volume_h5(Vol, "pre_merged_big_vol", "pre_merged_big_vol")
    '''
    '''
    Vol = np.zeros(Vol2.shape)
    
    Vol[np.where(Vol2 == 5108)] = 5108
    Vol[np.where(Vol2 == 5104)] = 5104
    Vol[np.where(Vol2 == 4585)] = 4585
    Vol[np.where(Vol2 == 5845)] = 5845
    '''
    '''
    Vol = torch.from_numpy(Vol)
    print("Finding Pairs")
    centers, fiber_ids, end_points, fiber_list = get_fiber_properties(Vol)

    print("Merging Neighbors")
    merge_inner_fibers(end_points, centers, fiber_ids, Vol, debug=1)


    tensors_io.save_volume_h5(Vol.numpy(), "merged_big_vol", "merged_big_vol")
    '''

    '''
    cube_size = 64
    X = np.load("../X_array.npy")
    Y = np.load("../300_labels.npy")
    final_pred = np.load("../final_pred.npy")
    final_pred = final_pred[0, 0, ...]
    Y = Y[0, 0:64, 0:64, 0:64]

    print("Num Labels Gt")
    print(len(np.unique(Y)))

    #clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05).fit(X)
    #labels = clust.labels_
    labels = DBSCAN(eps=0.35, min_samples=30).fit_predict(X)
    #labels = MeanShift(bandwidth=0.4).fit_predict(X)
    print("Num labels")
    original_labels = len(np.unique(labels))
    print(original_labels)

    print("Misclasified Pixels")
    print((len(np.where(labels == 1)[0])))


    object_indexes = np.where(final_pred.reshape(64 * 64 *64) > 0)
    space_labels = np.zeros([ 64 * 64 * 64, 1])
    space_labels[object_indexes] = torch.from_numpy(labels).unsqueeze(1) + 2
    space_labels = space_labels.reshape(cube_size, cube_size, cube_size)


    # space_labels = torch.from_numpy(space_labels)
    #space_labels = refine_watershed(space_labels, final_pred)

    '''
    '''
    post_watershed_labels = len(np.unique(labels))

    print("Post watershed Labels", post_watershed_labels)
    space_labels = space_labels.to(torch.device("cpu")).numpy()
    space_labels2 = np.copy(space_labels)


    pre_cleaned_list = fit_all_fibers_parallel(space_labels, np.array([0, 0, 0]))
    print(len(pre_cleaned_list))
    (fiber_list, volume_to_correct) = post_processing(pre_cleaned_list, space_labels)
    print(len(fiber_list))

    tensors_io.save_volume_h5(space_labels2)
    tensors_io.save_volume_h5(space_labels, "vol2", "vol2")
    evaluate_iou(space_labels, Y)
    tensors_io.save_volume_h5(Y, "vol3", "vol3")
    '''
    ''

