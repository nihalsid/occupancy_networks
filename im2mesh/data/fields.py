import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class CategoryField(Field):
    ''' Basic category field.'''
    def load(self, model_path, idx, category):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True


class ImagesField(Field):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        if self.random_view:
            idx_img = random.randint(0, len(files)-1)
        else:
            idx_img = 0
        filename = files[idx_img]

        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


class SDFPointField(Field):
    ''' SDF Field.

    It is the field used for loading sdfs.

    Args:
        file_path (str): path to sdf
        transform (list): list of transformations applied to loaded sdfs
    '''

    def __init__(self, num_samples=2048, sigma=5.0, transform=None, B_MIN=0, B_MAX=96):
        self.transform = transform
        self.B_MAX = B_MAX
        self.B_MIN = B_MIN
        self.sigma = sigma
        self.num_samples = num_samples

    def load_sdf_colors_file(self, model_path):
        trunc = 3.
        npzfile = np.load(model_path + ".npz")
        in_sdf, in_colors, tgt_sdf, tgt_colors = npzfile['in_sdf'], npzfile['in_colors'], npzfile['tgt_sdf'], npzfile['tgt_colors']
        mask_in_gt, mask_in_lt = np.greater(in_sdf, trunc), np.less(in_sdf, -trunc)
        in_sdf[mask_in_gt] = trunc
        in_colors[mask_in_gt] = 0
        in_sdf[mask_in_lt] = -trunc
        in_colors[mask_in_lt] = 0
        mask_tgt_gt, mask_tgt_lt = np.greater(tgt_sdf, trunc), np.less(tgt_sdf, -trunc)
        tgt_sdf[mask_tgt_gt] = trunc
        tgt_colors[mask_tgt_gt] = 0
        tgt_sdf[mask_tgt_lt] = -trunc
        tgt_colors[mask_tgt_lt] = 0
        return in_sdf, in_colors, tgt_sdf, tgt_colors,

    def do_sampling_from_sdf(self, in_sdf, tgt_sdf):
        surface_points_tgt = np.array(np.where(np.abs(tgt_sdf) < 1.0))
        sampled_idx = random.choices(list(range(surface_points_tgt.shape[1])), k=int(3 * self.num_samples))
        surface_points_tgt = surface_points_tgt[:, sampled_idx]
        surface_points_missing = np.array(np.where(np.logical_and(np.abs(tgt_sdf) < 1.0, np.logical_not(np.abs(in_sdf) < 1.0))))
        if surface_points_missing.shape[1] != 0:
            sampled_idx = random.choices(list(range(surface_points_missing.shape[1])), k=int(1 * self.num_samples))
            surface_points_missing = surface_points_missing[:, sampled_idx]
        surface_points = np.hstack((surface_points_tgt, surface_points_missing))

        sample_points = surface_points + np.random.normal(scale=self.sigma, size=surface_points.shape)
        length = self.B_MAX - self.B_MIN - 1
        random_points = (np.random.rand(4 * self.num_samples, 3) * length + self.B_MIN)
        sample_points = np.concatenate([sample_points.T, random_points], 0)
        out_of_bounds_pts = np.logical_or(
            np.logical_or(np.logical_or(sample_points[:, 0] < self.B_MIN, sample_points[:, 0] >= self.B_MAX - 1), np.logical_or(sample_points[:, 1] < self.B_MIN, sample_points[:, 1] >= self.B_MAX - 1)),
            np.logical_or(sample_points[:, 2] < self.B_MIN, sample_points[:, 2] >= self.B_MAX - 1))
        sample_points[out_of_bounds_pts, :] = (np.random.rand(3) * length + self.B_MIN)
        np.random.shuffle(sample_points)
        # sample_points[:, [0, 1, 2]] = sample_points[:, [2, 1, 0]]
        inside = np.zeros((sample_points.shape[0],), dtype=np.bool)
        sample_points_as_ints = sample_points.astype(np.int)
        mask = np.logical_and(np.logical_and(sample_points_as_ints[:, 0] > 0, sample_points_as_ints[:, 1] > 0), sample_points_as_ints[:, 2] > 0)
        mask = np.logical_and(mask, np.logical_and(np.logical_and(sample_points_as_ints[:, 0] < self.B_MAX, sample_points_as_ints[:, 1] < self.B_MAX), sample_points_as_ints[:, 2] < self.B_MAX))
        inside[mask] = np.abs(tgt_sdf[sample_points_as_ints[mask, 2], sample_points_as_ints[mask, 1], sample_points_as_ints[mask, 0]]) < 1

        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[:self.num_samples // 4] if nin > self.num_samples // 4 else inside_points
        nin = inside_points.shape[0]
        outside_points = outside_points[:(self.num_samples - nin)]

        samples = np.concatenate([inside_points, outside_points], 0) / self.B_MAX - 0.5
        labels = np.concatenate([np.ones((inside_points.shape[0])), np.zeros((outside_points.shape[0]))], 0)
        samples[:, [0, 1, 2]] = samples[:, [2, 1, 0]]

        # print('Number of samples in: ', nin, '/', nout)
        # want N x 3 and N
        return samples.astype(np.float32), labels.astype(np.float32)

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''

        in_sdf, in_colors, tgt_sdf, tgt_colors = self.load_sdf_colors_file(os.path.join(os.path.split(model_path)[0], "sdf", os.path.split(model_path)[1]))

        data = {
            None: in_sdf,
            'colors': 2 * np.transpose(in_colors, (3, 0, 1, 2)) / 255 - 1
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data


class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points,
            'normals': normals,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class ColoredPointCloudField(Field):
    ''' Colored Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''

    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        colors = 2 * pointcloud_dict['colors'].astype(np.float32) / 255 - 1

        data = {
            None: points,
            'colors': colors,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete

# NOTE: this will produce variable length output.
# You need to specify collate_fn to make it work with a data laoder
class MeshField(Field):
    ''' Mesh field.

    It provides the field used for mesh data. Note that, depending on the
    dataset, it produces variable length output, so that you need to specify
    collate_fn to make it work with a data loader.

    Args:
        file_name (str): file name
        transform (list): list of transforms applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'verts': mesh.vertices,
            'faces': mesh.faces,
        }

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
