import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import torch
import h5py
import os


#################### #################### Save Images #################### #################### ####################
def save_subvolume(V, path, scale=1, start=1, end=None):
    print('Saving Volumw at ' + path)
    if not os.path.isdir(path):
        os.mkdir(path)

    trans = transforms.ToPILImage()
    if(scale > 1):
        V = F.interpolate(V, scale_factor=scale, mode='nearest')
    if(len(V.size()) > 4):
        V = V[0, :, :, :, :]
    device = torch.device("cpu")
    V = V.to(device)
    size = V.size()

    if(end is None):
        end = size[-1]

    for i in range(start, end):
        img = V[:, :, :, i - start]
        img = trans(img)
        if(i  > 999):
            img.save(path + "/subV_" + str(i) + ".tif")
        elif(i  > 99):
            img.save(path + "/subV_0" + str(i) + ".tif")
        elif(i  > 9):
            img.save(path + "/subV_00" + str(i) + ".tif")
        else:
            img.save(path + "/subV_000" + str(i ) + ".tif")


def save_subvolume_instances(V, M, path,  start=0, end=None):
    print('Saving Instances at ' + path)
    if not os.path.isdir(path):
        os.mkdir(path)

    trans = transforms.ToPILImage()
    # V is initially as channels x rows x cols x slices
    # Make it slices x rows x cols x channels
    # V = V.permute(3, 1, 2, 0)
    if(len(V.size()) > 4):
        V = V[0, :, :, :, :]

    if(len(M.size()) > 4):
        M = M[0, :, :, :, :]

    device = torch.device("cpu")
    V = V.to(device)
    M = M.to(device)
    num_classes = M.max().int().item()
    colors_r = torch.rand(num_classes)
    colors_g = torch.rand(num_classes)
    colors_b = torch.rand(num_classes)
    
    if(end is None):
        end = V.shape[-1]
    for i in range(start, end):
        img = V[:, :, :, i - start]
        mask = M[:, :, :, i - start]

        overlay = torch.cat([img, img, img], 0)
        for c in mask.unique():
            if(c == 0):
                continue
            indxs = (mask[0, :, :] == c).nonzero()
            if(c == 1):
                for idx in indxs:
                    overlay[0, idx[0], idx[1]] = 1
                    overlay[1, idx[0], idx[1]] = 1
                    overlay[2, idx[0], idx[1]] = 1
            else:
                for idx in indxs:
                    overlay[0, idx[0], idx[1]] = colors_r[c - 1]
                    overlay[1, idx[0], idx[1]] = colors_g[c - 1]
                    overlay[2, idx[0], idx[1]] = colors_b[c - 1]

        # overlay[0, :, :] += 2 * mask[0, :, :].clamp(0, 1)
        # overlay[1, :, :] += 2 * (mask[0, :, :] - 1).clamp(0, 1)
        # overlay = overlay.clamp(min_v, max_v)
        img = trans(overlay)

        if(i > 999):
            img.save(path + "/subV_" + str(i) + ".tif")
        elif(i > 99):
            img.save(path + "/subV_0" + str(i) + ".tif")
        elif(i > 9):
            img.save(path + "/subV_00" + str(i) + ".tif")
        else:
            img.save(path + "/subV_000" + str(i) + ".tif")


def save_subvolume_color(V, M, path, num_classes=3, scale=1, start=1, end=None):
    if not os.path.isdir(path):
        os.mkdir(path)

    # resize_vol = transforms.Resize([450, 450], interpolation=2)
    # resize_mask = transforms.Resize([450,450], interpolation=Image.NEAREST)
    if(scale > 1):
        V = F.interpolate(V, scale_factor=scale, mode='trilinear', align_corners=True)
        M = F.interpolate(M, scale_factor=scale, mode='nearest')

    trans = transforms.ToPILImage()
    # V is initially as channels x rows x cols x slices
    # Make it slices x rows x cols x channels
    # V = V.permute(3, 1, 2, 0)
    if(len(V.size()) > 4):
        V = V[0, :, :, :, :]

    if(len(M.size()) > 4):
        M = M[0, :, :, :, :]

    device = torch.device("cpu")
    V = V.to(device)
    M = M.to(device)
    size = V.size()
    colors_r = [  0, 0.5, 1,   0,  0.5,  1,   0, 0.5, 1,   1]
    colors_g = [1 , 0.5,   0,  1, 1,  1, 0.5, 0.5,   0,   1]
    colors_b = [  0, 1, 1,  0.5,    0,   0,   0,   0,   0, 1]
    #    colors_r = [  0  ,  0,   0,  0,    0,  0,     0.5, 0.5,  0.5,    0.5]
    #    colors_g = [  0  ,  0, 0.5,  1,    1,  1,     0.5,   1,    1,      1]
    #    colors_b = [  0.5,  1, 0.5,  0,  0.5,  1,     0.5,   0,  0.5,      1]
    if(end is None):
        end = size[-1] - 1

    for i in range(start, end + 1):
        img = V[:, :, :, i - start]
        mask = M[:, :, :, i - start]
        # img = resize_vol(img)
        # mask = resize_mask(img)
        overlay = torch.cat([img, img, img], 0)
        for c in range(1, num_classes + 1):
            indxs = (mask[0, :, :] == c).nonzero()
            for idx in indxs:
                overlay[0, idx[0], idx[1]] = colors_r[c-1]
                overlay[1, idx[0], idx[1]] = colors_g[c-1]
                overlay[2, idx[0], idx[1]] = colors_b[c-1]

        # overlay[0, :, :] += 2 * mask[0, :, :].clamp(0, 1)
        # overlay[1, :, :] += 2 * (mask[0, :, :] - 1).clamp(0, 1)
        # overlay = overlay.clamp(min_v, max_v)
        img = trans(overlay)

        if(i > 999):
            img.save(path + "/subV_" + str(i) + ".tif")
        elif(i > 99):
            img.save(path + "/subV_0" + str(i) + ".tif")
        elif(i > 9):
            img.save(path + "/subV_00" + str(i) + ".tif")
        else:
            img.save(path + "/subV_000" + str(i) + ".tif")


def load_full_volume(path, start=0, end=450, scale=2):
    list_of_names = sorted(os.listdir(path))
    num_Z = len(list_of_names)
    # Update End Value
    end = min(end, num_Z)
    print("Reading image indexes: " + str(start) + ":" + str(end) + "/" + str(num_Z))
    # Read and crop first image
    im = Image.open(path + '/' + list_of_names[0])
    im = torch.from_numpy(np.array(im, np.int16, copy=False)).unsqueeze(0)
    im = im[:, 200:-311, 280:-231]
    im = im[:, ::scale, ::scale]
    size = im.size()
    V = torch.zeros(size[0], size[1], size[2], (end - start + 1) // scale)
    countZ = 0

    # Read and crop all images
    for i in range(start, end + 1, scale):
        name = list_of_names[i]
        if name[-1] == 'f':
            im = Image.open(path + '/' + name)
            im = np.asarray(im, np.uint16)
            im = im[200:-311, 280:-231]
            im = im.astype(np.float)
            im = im / 2**16
            # im = im.astype(np.float)
            im = torch.from_numpy(im).unsqueeze(0)
            im = im[:, ::scale, ::scale]
            V[:, :, :, countZ] = im
            countZ += 1
    return V


def load_volume_uint16(path, start=0, end=449, scale=2):
    print('Loading Instances from ' + path)
    list_of_names = sorted(os.listdir(path))
    num_Z = len(list_of_names)
    # Update End Value
    end = min(end, num_Z - 1)

    # Read and crop first image
    im = Image.open(path + '/' + list_of_names[0])
    im = torch.from_numpy(np.array(im, np.int16, copy=False)).unsqueeze(0)
    im = im[:, ::scale, ::scale]
    size = im.size()
    V = torch.zeros(size[0], size[1], size[2], (end - start + 1) // scale)
    countZ = 0
    # Read and crop all images
    for i in range(start, end , scale):
        name = list_of_names[i]
        if name[-1] == 'f':
            im = Image.open(path + '/' + name)
            im = np.asarray(im, np.uint16)
            im = im.astype(np.float)
            # im = im / 2**16
            # im = im.astype(np.float)
            im = torch.from_numpy(im).unsqueeze(0)
            im = im[:, ::scale, ::scale]
            V[:, :, :, countZ] = im
            countZ += 1
    return V


def load_volume(path, scale=2):
    print('Loading Volume from ' + path)
    make_tensor = transforms.ToTensor()
    list_of_names = sorted(os.listdir(path))
    num_Z = len(list_of_names)
    im = Image.open(path + '/' + list_of_names[0])
    im = make_tensor(im)
    im = im[:, ::scale, ::scale]
    size = im.size()
    V = torch.zeros(size[0], size[1], size[2], num_Z // scale)
    countZ = 0
    for i in range(0, len(list_of_names) - 1, scale):
        name = list_of_names[i]
        if name[-1] == 'f':
            im = Image.open(path + '/' + name)
            im = make_tensor(im)
            im = im[:, ::scale, ::scale]
            V[:, :, :, countZ] = im
            countZ += 1
    return V

#################### #################### H5 Files #################### #################### ####################

def save_volume_h5(V, name='Volume', dataset_name='Volume', directory='./h5_files'):
    if not os.path.isdir(directory):
        os.mkdir(directory)

    with h5py.File(directory + "/" + name + '.h5', 'w') as f:
        dset = f.create_dataset(dataset_name, data=V, maxshape=(V.shape[0], V.shape[1], None))

    create_xml_file(V.shape, directory, name, dataset_name)


def append_volume_h5(V, name='Volume', dataset_name='Volume', directory='./h5_files'):
   
    with h5py.File(directory + "/" + name + ".h5", 'a') as f:
        V_rows, V_cols, V_depth  = V.shape
        old_rows, old_cols, old_depth = f[dataset_name].shape

        if(V_rows != old_rows or V_cols != old_cols):
            print("Dataset Append Error: Dimensions must match")
            return

        new_depth = old_depth + V_depth

        f[dataset_name].resize(new_depth, axis = 2)
        f[dataset_name][:, :, -V_depth:] = V

    new_dims = [old_rows, old_cols, new_depth]
    create_xml_file(new_dims, directory, name, dataset_name)


def create_xml_file(volume_shape, directory, name, dataset_name):
    Nx, Ny, Nz = volume_shape
    xml_filename = directory + "/" + name + ".xmf"
    f = open(xml_filename, 'w')
    # Header for xml file
    f.write('''<?xml version="1.0" ?>
            <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
            <Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
            <Domain>
            ''')

    #  Naming datasets
    dataSetName1 = name +'.h5:/' + dataset_name

    f.write('''
                <Grid Name="Box Test" GridType="Uniform"> #
                <Topology TopologyType="3DCORECTMesh" Dimensions="%d %d %d"/>
                <Geometry GeometryType="ORIGIN_DXDYDZ">
                <DataItem Name="Origin" DataType="Float" Dimensions="3" Format="XML">0.0 0.0 0.0</DataItem>
                <DataItem Name="Spacing" DataType="Float" Dimensions="3" Format="XML">1.0 1.0 1.0</DataItem>
                </Geometry>
                ''' % (Nx, Ny, Nz))

    f.write('''\n
                <Attribute Name="S" AttributeType="Scalar" Center="Node">
                <DataItem Format="HDF" Dimensions="%d %d %d" NumberType="UInt" Precision="2"
                >%s
                </DataItem>
                </Attribute>
                ''' % (Nx, Ny, Nz, dataSetName1))

    # End the xmf file
    f.write('''
       </Grid>
    </Domain>
    </Xdmf>
    ''')

    f.close()


def read_volume_h5(name='Volume', dataset_name='Volume', directory='./h5_files'):
    filename = directory + "/" + name + ".h5"
    f = h5py.File(filename, 'r')
    a_group_key = list(f.keys())[0]
    data = f[a_group_key][()]
    return data


#################### #################### Cropping Utils #################### #################### ####################

def random_crop_3D_image_batched(img, mask, crop_size):
    if(type(crop_size) not in (tuple, list)):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[4]:
        lb_z = np.random.randint(0, img.shape[4] - crop_size[2])
    elif crop_size[2] == img.shape[4]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")
    V = img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
    Mask = mask[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]
    return (V, Mask)


def full_crop_3D_image_batched(img, mask, lb_x, lb_y, lb_z, crop_size):
    V = img[:, :, lb_x:lb_x + crop_size, lb_y:lb_y + crop_size, lb_z:lb_z + crop_size]
    Mask = mask[:, :, lb_x:lb_x + crop_size, lb_y:lb_y + crop_size, lb_z:lb_z + crop_size]
    return (V, Mask)


#################### #################### Filters #################### #################### ####################

def create_histogram_ref(reference):
    reference = reference.numpy()
    tmpl_values, tmpl_counts = np.unique(reference.ravel(), return_counts=True)
    tmpl_quantiles = np.cumsum(tmpl_counts).astype(np.float) / (reference.size)
    print("Creating Reference...")
    reference_hist = [tmpl_values, tmpl_quantiles]
    with open('info_files/histogram_reference.pickle', 'wb') as f:
        pickle.dump(reference_hist, f)

    return (reference_hist)



def clean_noise(vol, data_path):
    '''
        vol must be tensor of shape [height, width, depth]
    '''
    vol = vol.numpy()
    clean_vol = np.zeros(vol.shape)

    try:
        tmpl_values, tmpl_quantiles =  pickle.load( open('info_files/histogram_reference.pickle', "rb" ) )
    except:
        # Keep preset values
        print("WARNING: DID NOT FIND REFERENCE HISTOGRAM...RESULTS MAY LOOK UGLY")
        reference =  load_full_volume(data_path, 0 ,5)
        tmpl_values, tmpl_quantiles = create_histogram_ref(reference[0, ...])    
    
    print("Adapting Sample for Neural Net")
    for i in range(vol.shape[2]):
        source = vol[..., i]

        src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)

        src_quantiles = np.cumsum(src_counts).astype(np.float) / (source.size)
        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        matched = interp_a_values[src_unique_indices].reshape(source.shape)
        clean_vol[..., i] = matched
    clean_vol = torch.tensor(clean_vol)
    return clean_vol

def cylinder_filter(data_volume_shape, center=None, radius=None):
    [rows, cols, slices] = data_volume_shape
    grid = np.mgrid[[slice(i) for i in [rows, cols]]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid)**2, 0))
    res = (phi > 0).astype(np.float)
    res = np.repeat(res[:, :, np.newaxis], slices , axis=2)
    return res

if __name__ == '__main__':
    from scipy import misc
    # a = misc.imresize(a, 2, interp='nearest')
    
    data_path = 'TRAINING_DATA'
    # mask_path = 'TRAINING_LABELS'
    mask_path = 'fibers_uint16'
    final_pred = read_volume_h5(name='merged_labels_final', dataset_name='merged_labels_final', directory='./h5_development')
    masks = load_volume_uint16(mask_path, scale=2).long()
    
    # final_pred = ndimage.zoom(final_pred, 2,  mode='nearest', prefilter=False)
    
    final_pred = torch.from_numpy(final_pred)
    final_pred = final_pred.unsqueeze(0)

    data_volume = load_volume(data_path, scale=2)
    data_volume[0, ...] =  clean_noise(data_volume[0, ...], data_path)

    # save_subvolume_instances(data_volume,final_pred, 'labels_presentation_augsut_20')
    
    save_subvolume_instances(data_volume,masks, 'training_data_presentation_augsut_20')
    
################################################## Embedded Files ####################################    
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

    unique_labels = torch.unique(labeled_pixels)
    N_objects = len(unique_labels)
    mu_vector = torch.zeros(N_objects, N_embedded)

    for c in range(N_objects):
        fiber_id = unique_labels[c]
        idx_c = (labeled_pixels == fiber_id).nonzero()

        # xi vector
        x_i = torch.gather(embeddings, 0, idx_c.repeat(1, N_embedded))

        # get mu
        mu = x_i.mean(0)
        mu_vector[c, :] = mu
    resta = mu_vector[c, :] - x_i.cpu()
    lv_term = torch.norm(resta, 2) - 0

    lv_temp2 = torch.norm(resta, 2, dim=1)
    ff = 0
    for k in range(x_i.shape[0]):
        ff += torch.norm(resta[k, :], 2)

    mu_vector = mu_vector.detach().cpu().numpy()
    # Y = tsne(object_pixels[::4, :].detach().numpy(), num_display_dims, num_dims, tsne_lr)
    # Y = pca(object_pixels.detach().numpy(), no_dims=2).real
    from tsnecuda import TSNE
    # from tsne import tsne
    X = object_pixels.cpu().detach().numpy()
    Y = TSNE(n_components=2, perplexity=20, learning_rate=50).fit_transform(X)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(Y[:, 0], Y[:, 1], 5, labeled_pixels, cmap='tab20b')
    if iteration is None:
        iteration = 0
    plt.savefig("low_dim_embeeding/embedded_%d.png" % iteration)
    plt.close(fig)
    # print(X[:, 0].shape)

    # Y = tsne(X, 2, 32, 20.0)
    # fig = pylab.figure()
    # pylab.scatter(Y[:, 0], Y[:, 1], 5, labeled_pixels, cmap='tab20b')
    # if iteration is None:labels
    #    iteration = 0
    # pylab.savefig("low_dim_embeeding/embedded_%d.png" % iteration)
    # pylab.close(fig)
    '''
    ax1 = fig.add_subplot(222)
    ax1.scatter(Y[:, 0], Y[:, 2], 5, labeled_pixels, cmap='tab20b')
    ax1 = fig.add_subplot(223)
    ax1.scatter(Y[:, 0], Y[:, 3], 5, labeled_pixels, cmap='tab20b')
    ax1 = fig.add_subplot(224)
    ax1.scatter(Y[:, 0], Y[:, 4], 5, labeled_pixels, cmap='tab20b')
    '''
    fig = pylab.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(X[:, 0], X[:, 1], 5, labeled_pixels, cmap='tab20b')
    ax1.scatter(mu_vector[:, 0], mu_vector[:, 1], 40, unique_labels, cmap='tab20b', marker="x")
    ax1 = fig.add_subplot(222)
    ax1.scatter(X[:, 0], X[:, 2], 5, labeled_pixels, cmap='tab20b')
    ax1.scatter(mu_vector[:, 0], mu_vector[:, 2], 40, unique_labels, cmap='tab20b', marker="x")
    ax1 = fig.add_subplot(223)
    ax1.scatter(X[:, 0], X[:, 3], 5, labeled_pixels, cmap='tab20b')
    ax1.scatter(mu_vector[:, 0], mu_vector[:, 3], 40, unique_labels, cmap='tab20b', marker="x")
    ax1 = fig.add_subplot(224)
    ax1.scatter(X[:, 0], X[:, 4], 5, labeled_pixels, cmap='tab20b')
    ax1.scatter(mu_vector[:, 0], mu_vector[:, 4], 40, unique_labels, cmap='tab20b', marker="x")
    plt.savefig("2_embedding/embedded_%d.png" % iteration)
    plt.close(fig)

    if detected_labels is not None:
        pylab.scatter(Y[:, 0], Y[:, 1], 5, detected_labels, cmap='tab20b')
        pylab.savefig("low_dim_embeeding/embedded_%d.png" % (iteration + 1000))
        pylab.close(fig)