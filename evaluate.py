import os
import json
import shutil
import trimesh
import numpy as np
import cadquery as cq
from tqdm import tqdm
from functools import partial
from scipy.spatial import cKDTree
from collections import defaultdict
from argparse import ArgumentParser
from multiprocessing import Process
from multiprocessing.pool import Pool

import open3d


class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super(NonDaemonPool, self).Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc


def sample_mesh_points(mesh, n_points):
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def compute_chamfer_distance_points(gt_points, pred_points):
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))


def compute_chamfer_distance(gt_mesh, pred_mesh, n_points):
    gt_points = sample_mesh_points(gt_mesh, n_points)
    pred_points = sample_mesh_points(pred_mesh, n_points)
    return compute_chamfer_distance_points(gt_points, pred_points)


def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume
        
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except:
        pass


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def py_file_to_mesh_and_brep_files(py_path, mesh_path, brep_path):
    try:
        with open(py_path, 'r') as f:
            py_string = f.read()
        exec(py_string, globals())
        compound = globals()['r'].val()
        mesh = compound_to_mesh(compound)
        assert len(mesh.faces) > 2
        mesh.export(mesh_path)
        # comment this line if no need to export brep
        # cq.exporters.export(compound, brep_path)
    except:
        pass


def py_file_to_mesh_and_brep_files_safe(py_path, mesh_path, brep_path):
    process = Process(
        target=py_file_to_mesh_and_brep_files,
        args=(py_path, mesh_path, brep_path))
    process.start()
    process.join(3)

    if process.is_alive():
        print('process alive:', py_path)
        process.terminate()
        process.join()


def run_cd_single(py_file_name, pred_py_path, pred_mesh_path, pred_brep_path, gt_path,
                  n_points, gt_format, point_cloud_exts, mesh_ext):
    eval_file_name = py_file_name[:py_file_name.rfind('+')]
    py_path = os.path.join(pred_py_path, py_file_name)
    mesh_path = os.path.join(pred_mesh_path, py_file_name[:-3] + '.stl')
    brep_path = os.path.join(pred_brep_path, py_file_name[:-3] + '.step')
    py_file_to_mesh_and_brep_files_safe(py_path, mesh_path, brep_path)

    cd, iou = None, None
    try:  # apply_transform fails for some reason; or mesh path can not exist
        pred_mesh = trimesh.load_mesh(mesh_path)
        center = (pred_mesh.bounds[0] + pred_mesh.bounds[1]) / 2.0
        pred_mesh.apply_translation(-center)
        extent = np.max(pred_mesh.extents)
        if extent > 1e-7:
            pred_mesh.apply_scale(1.0 / extent)
        pred_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
        if gt_format == 'mesh':
            gt_mesh = trimesh.load_mesh(resolve_gt_mesh_path(gt_path, eval_file_name, mesh_ext))
            cd = compute_chamfer_distance(gt_mesh, pred_mesh, n_points)
            iou = compute_iou(gt_mesh, pred_mesh)
        else:
            gt_points = load_point_cloud(resolve_gt_point_cloud_path(gt_path, eval_file_name, point_cloud_exts))
            pred_points = sample_mesh_points(pred_mesh, n_points)
            cd = compute_chamfer_distance_points(gt_points, pred_points)
    except:
        pass
    
    index = py_file_name[len(eval_file_name) + 1: -3]
    return dict(file_name=eval_file_name, id=index, cd=cd, iou=iou)


def run(gt_path, pred_py_path, n_points, gt_format, point_cloud_exts, mesh_ext):
    pred_mesh_path = os.path.join(os.path.dirname(pred_py_path), 'tmp_mesh')
    pred_brep_path = os.path.join(os.path.dirname(pred_py_path), 'tmp_brep')
    best_names_path = os.path.join(os.path.dirname(pred_py_path), 'tmp.txt')
    metrics_path = os.path.join(os.path.dirname(pred_py_path), 'metrics.json')

    # should be no predicted meshes from previous experiments
    if os.path.exists(pred_mesh_path):
        shutil.rmtree(pred_mesh_path)
    if os.path.exists(pred_brep_path):
        shutil.rmtree(pred_brep_path)

    os.makedirs(pred_mesh_path, exist_ok=True)
    os.makedirs(pred_brep_path, exist_ok=True)

    # compute chamfer distance and iou for each sample
    py_file_names = os.listdir(pred_py_path)
    with NonDaemonPool(16) as pool:
        py_metrics = list(tqdm(pool.imap(
            partial(
                run_cd_single,
                pred_py_path=pred_py_path,
                pred_mesh_path=pred_mesh_path,
                pred_brep_path=pred_brep_path,
                gt_path=gt_path,
                n_points=n_points,
                gt_format=gt_format,
                point_cloud_exts=point_cloud_exts,
                mesh_ext=mesh_ext),
            py_file_names), total=len(py_file_names)))

    # aggregate metrics per eval_file_name
    metrics = defaultdict(lambda: defaultdict(list))
    for m in py_metrics:
        if m['cd'] is not None:
            metrics[m['file_name']]['cd'].append(m['cd'])
            metrics[m['file_name']]['id'].append(m['id'])
        if m['iou'] is not None:
            metrics[m['file_name']]['iou'].append(m['iou'])

        # empty value for invalid predictions
        metrics[m['file_name']]

    
    # select best metrics per eval_file_name
    ir_cd, ir_iou, cd, iou, best_names = 0, 0, list(), list(), list()
    for key, value in metrics.items():
        if len(value['cd']):
            argmin = np.argmin(value['cd'])
            cd.append(value['cd'][argmin])
            index = value['id'][argmin]
            best_names.append(f'{key}+{index}.py')
        else:
            ir_cd += 1

        if len(value['iou']):
            iou.append(np.max(value['iou']))
        else:
            ir_iou += 1

    with open(best_names_path, 'w') as f:
        f.writelines([line + '\n' for line in best_names])

    summary = {
        'invalid_cd': ir_cd,
        'invalid_iou': ir_iou,
        'mean_iou': None,
        'median_cd': None,
        'skip_stats': list(),
    }

    if len(iou):
        summary['mean_iou'] = float(np.mean(iou))
        print(f"mean iou: {summary['mean_iou']:.3f}", end=' ')
    else:
        print('mean iou: N/A', end=' ')

    if len(cd):
        summary['median_cd'] = float(np.median(cd))
        print(f"median cd: {summary['median_cd'] * 1000:.3f}")
    else:
        print('median cd: N/A')

    cd = sorted(cd)
    for i in range(5):
        if len(metrics):
            skip_mean_cd = float(np.mean(cd[:len(cd) - i])) if len(cd) - i > 0 else None
            invalid_ratio = float((ir_cd + i) / len(metrics))
        else:
            skip_mean_cd = None
            invalid_ratio = None

        summary['skip_stats'].append({
            'skip': i,
            'invalid_ratio': invalid_ratio,
            'mean_cd': skip_mean_cd,
        })

        if skip_mean_cd is not None and invalid_ratio is not None:
            print(f'skip: {i} ir: {invalid_ratio * 100:.2f}',
                  f'mean cd: {skip_mean_cd * 1000:.3f}')
        else:
            print(f'skip: {i} ir: N/A mean cd: N/A')

    metrics_serialized = {
        key: {
            'cd': [float(v) for v in value['cd']],
            'iou': [float(v) for v in value['iou']],
            'id': list(value['id']),
        }
        for key, value in metrics.items()
    }

    results = {
        'summary': summary,
        'best_names': best_names,
        'metrics': metrics_serialized,
    }

    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)


# To overcome CadQuery memory leaks, we call each exec() in a separate Process with
# timeout of 3 seconds. The Pool is tweaked to support non-daemon processes that can
# call one more nested process.
def resolve_gt_mesh_path(gt_root, file_name, mesh_ext):
    path = os.path.join(gt_root, f'{file_name}.{mesh_ext}')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Ground truth mesh {path} not found')
    return path


def resolve_gt_point_cloud_path(gt_root, file_name, point_cloud_exts):
    for ext in point_cloud_exts:
        candidate = os.path.join(gt_root, f'{file_name}.{ext}')
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f'No ground truth point cloud found for {file_name} with extensions {point_cloud_exts}')


def load_point_cloud(path):
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt-path', type=str, default='./data/deepcad_test_mesh')
    parser.add_argument('--gt-format', type=str, choices=['mesh', 'point_cloud'], default='mesh')
    parser.add_argument('--gt-point-cloud-exts', type=str, default='ply,pcd,xyz,txt,npz,npy')
    parser.add_argument('--gt-mesh-ext', type=str, default='stl')
    parser.add_argument('--pred-py-path', type=str, default='./work_dirs/tmp_py')
    parser.add_argument('--n-points', type=int, default=8192)
    args = parser.parse_args()
    run(
        args.gt_path,
        args.pred_py_path,
        args.n_points,
        args.gt_format,
        tuple(ext.strip().lower() for ext in args.gt_point_cloud_exts.split(',') if ext.strip()),
        args.gt_mesh_ext.lower())
