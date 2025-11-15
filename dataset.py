import os
import pickle
import open3d
import trimesh
import skimage
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points


def mesh_to_point_cloud(mesh, n_points, n_pre_points=8192):
    vertices, faces = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    vertices = vertices[ids]
    return np.asarray(vertices)


def mesh_to_image(mesh, camera_distance=-1.8, front=[1, 1, 1], width=500, height=500, img_size=128):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    front_array = np.array(front, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    
    eye = lookat + front_array * camera_distance
    right = np.cross(up, front_array)
    right /= np.linalg.norm(right)
    true_up = np.cross(front_array, right)
    rotation_matrix = np.column_stack((right, true_up, front_array)).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = -rotation_matrix @ eye

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = skimage.transform.resize(
        image,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True).astype(np.uint8)

    return Image.fromarray(image)


class CadRecodeDataset(Dataset):
    def __init__(self, root_dir, split, n_points, normalize_std_pc, noise_scale_pc, img_size,
                normalize_std_img, noise_scale_img, num_imgs, mode, n_samples=None, ext='stl',
                input_source='mesh', point_cloud_exts=None, image_exts=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.n_samples = n_samples
        self.n_points = n_points
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.normalize_std_img = normalize_std_img
        self.noise_scale_img = noise_scale_img
        self.num_imgs = num_imgs
        self.mode = mode
        self.input_source = input_source
        self.ext = ext.lower()
        self.point_cloud_exts = tuple(
            ext.lower() for ext in (point_cloud_exts or ['ply', 'pcd', 'xyz', 'txt', 'npz', 'npy']))
        self.image_exts = tuple(
            ext.lower() for ext in (image_exts or ['png', 'jpg', 'jpeg', 'bmp']))
        if self.split in ['train', 'val']:
            pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
            with open(pkl_path, 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            split_root = os.path.join(self.root_dir, self.split)
            self.annotations = self._build_inference_annotations(split_root)

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        if self.mode == 'pc':
            input_item = self.get_point_cloud(item)
        elif self.mode == 'img':
            input_item = self.get_img(item)
        elif self.mode == 'pc_img':
            if np.random.rand() < 0.5:
                input_item = self.get_point_cloud(item)
            else:
                input_item = self.get_img(item)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        input_item['file_name'] = self._resolve_file_name(item)

        if self.split in ['train', 'val']:
            py_path = item['py_path']
            py_path = os.path.join(self.root_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer

        return input_item

    def get_img(self, item):
        if 'mesh_path' in item:
            mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
            if self.split in ['train', 'val']:
                mesh.apply_transform(trimesh.transformations.scale_matrix(1 / self.normalize_std_img))
                mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            mesh = open3d.geometry.TriangleMesh()
            mesh.vertices = open3d.utility.Vector3dVector(vertices)
            mesh.triangles = open3d.utility.Vector3iVector(faces)
            mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
            mesh.compute_vertex_normals()

            fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
            images = []
            for front in fronts:
                image = mesh_to_image(
                    mesh, camera_distance=-0.9, front=front, img_size=self.img_size)
                images.append(image)
        else:
            images = []
            for rel_path in item['image_paths']:
                image = Image.open(os.path.join(self.root_dir, rel_path)).convert('RGB')
                if self.img_size is not None:
                    image = image.resize((self.img_size, self.img_size))
                images.append(image)

        if len(images) == 0:
            raise ValueError('No images found for the provided sample')

        images = [ImageOps.expand(image, border=3, fill='black') for image in images]
        if self.num_imgs == 1:
            images = [images[0]]
        elif self.num_imgs == 2:
            if len(images) < 2:
                raise ValueError('Expected at least 2 images for num_imgs=2 input mode')
            images = [Image.fromarray(np.hstack((
                np.array(images[0]), np.array(images[1])
            )))]
        elif self.num_imgs == 4:
            if len(images) < 4:
                raise ValueError('Expected at least 4 images for num_imgs=4 input mode')
            images = [Image.fromarray(np.vstack((
                np.hstack((np.array(images[0]), np.array(images[1]))),
                np.hstack((np.array(images[2]), np.array(images[3])))
            )))]
        else:
            raise ValueError(f'Invalid number of images: {self.num_imgs}')

        input_item = {
            'video': images,
            'description': 'Generate cadquery code'
        }
        return input_item

    def get_point_cloud(self, item):
        if 'mesh_path' in item:
            mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
            mesh = self._augment_pc(mesh)
            point_cloud = mesh_to_point_cloud(mesh, self.n_points)

            if self.split in ['train', 'val']:
                point_cloud = point_cloud / self.normalize_std_pc
            else:
                point_cloud = (point_cloud - 0.5) * 2
        else:
            point_cloud = self._load_point_cloud(os.path.join(self.root_dir, item['point_cloud_path']))
            point_cloud = self._resample_point_cloud(point_cloud, self.n_points)
            point_cloud = self._normalize_external_point_cloud(point_cloud)

        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        return input_item

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh

    def _build_inference_annotations(self, split_root):
        if self.input_source == 'mesh':
            paths = sorted(os.listdir(split_root))
            return [
                {'mesh_path': os.path.join(self.split, f)}
                for f in paths if f.lower().endswith(f'.{self.ext}')
            ]

        if self.input_source == 'point_cloud':
            paths = sorted(os.listdir(split_root))
            annotations = []
            for f in paths:
                if any(f.lower().endswith(f'.{ext}') for ext in self.point_cloud_exts):
                    annotations.append({'point_cloud_path': os.path.join(self.split, f)})
            return annotations

        if self.input_source == 'multi_view':
            dirs = [d for d in sorted(os.listdir(split_root)) if os.path.isdir(os.path.join(split_root, d))]
            annotations = []
            for d in dirs:
                dir_root = os.path.join(split_root, d)
                image_paths = [
                    os.path.join(self.split, d, f)
                    for f in sorted(os.listdir(dir_root))
                    if any(f.lower().endswith(f'.{ext}') for ext in self.image_exts)
                ]
                if image_paths:
                    annotations.append({'image_paths': image_paths, 'sample_dir': d})
            return annotations

        raise ValueError(f'Unknown input_source {self.input_source}')

    def _resolve_file_name(self, item):
        if 'mesh_path' in item:
            return os.path.basename(item['mesh_path'])[:-4]
        if 'point_cloud_path' in item:
            return os.path.splitext(os.path.basename(item['point_cloud_path']))[0]
        if 'sample_dir' in item:
            return os.path.basename(item['sample_dir'])
        raise ValueError('Unable to resolve file name for dataset item')

    def _load_point_cloud(self, path):
        if path.lower().endswith('.npz'):
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                for key in ('points', 'point_cloud', 'pc'):
                    if key in data:
                        return np.asarray(data[key])
                return np.asarray(list(data.values())[0])
            return np.asarray(data)
        if path.lower().endswith('.npy'):
            return np.load(path)
        if path.lower().endswith(('.txt', '.xyz')):
            return np.loadtxt(path, dtype=np.float32)

        point_cloud = open3d.io.read_point_cloud(path)
        if point_cloud is None:
            raise ValueError(f'Unable to read point cloud from {path}')
        return np.asarray(point_cloud.points)

    def _resample_point_cloud(self, points, n_points):
        if points.shape[0] == 0:
            raise ValueError('Point cloud is empty')
        if points.shape[0] >= n_points:
            _, ids = sample_farthest_points(torch.tensor(points).unsqueeze(0), K=n_points)
            ids = ids[0].numpy()
            return points[ids]
        # duplicate points if there are fewer than required
        repeat = int(np.ceil(n_points / points.shape[0]))
        points = np.tile(points, (repeat, 1))
        points = points[:n_points]
        return points

    def _normalize_external_point_cloud(self, points):
        centroid = np.mean(points, axis=0, keepdims=True)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale
        return points


class Text2CADDataset(Dataset):
    def __init__(self, root_dir, split, code_dir='cadquery', n_samples=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_samples = n_samples
        self.code_dir = code_dir
        pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        input_item = {
            'description': item['description'],
            'file_name': item['uid']
        }

        if self.split in ['train', 'val']:
            py_path = f'{item["uid"]}.py'
            py_path = os.path.join(self.root_dir, self.code_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer
        return input_item
