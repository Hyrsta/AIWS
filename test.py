import os
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser

import torch
from transformers import AutoProcessor
from torch.utils.data import DataLoader, ConcatDataset

from cadrille import Cadrille, collate
from dataset import Text2CADDataset, CadRecodeDataset


def run(data_path, split, mode, checkpoint_path, py_path, input_source,
        point_cloud_exts, image_exts, mesh_ext):
    # should be no predicted codes from previous experiments
    os.makedirs(args.py_path, exist_ok=True)
    assert len(os.listdir(py_path)) == 0

    model = Cadrille.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map='auto')

    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct', 
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28,
        padding_side='left')

    if mode == 'text':
        dataset = Text2CADDataset(
            root_dir=os.path.join(data_path, 'text2cad'),
            split='test')
        batch_size = 32
    else:  # mode in ('pc', 'img')
        if mode == 'pc' and input_source == 'multi_view':
            raise ValueError('Multi-view images are incompatible with point cloud mode')
        if mode == 'img' and input_source == 'point_cloud':
            raise ValueError('Point cloud files are incompatible with image mode')
        dataset = CadRecodeDataset(
            root_dir=data_path,
            split=split,
            n_points=256,
            normalize_std_pc=100,
            noise_scale_pc=None,
            img_size=128,
            normalize_std_img=200,
            noise_scale_img=-1,
            num_imgs=4,
            mode=mode,
            ext=mesh_ext,
            input_source=input_source,
            point_cloud_exts=point_cloud_exts,
            image_exts=image_exts)
        batch_size = 256

    n_samples = 1
    counter = 0
    dataloader = DataLoader(
        dataset=ConcatDataset([dataset] * n_samples),
        batch_size=batch_size,
        num_workers=16,
        collate_fn=partial(collate, processor=processor, n_points=256, eval=True))

    for batch in tqdm(dataloader):
        generated_ids = model.generate(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            point_clouds=batch['point_clouds'].to(model.device),
            is_pc=batch['is_pc'].to(model.device),
            is_img=batch['is_img'].to(model.device),
            pixel_values_videos=batch['pixel_values_videos'].to(model.device) if batch.get('pixel_values_videos', None) is not None else None,
            video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get('video_grid_thw', None) is not None else None,
            max_new_tokens=768)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
        ]
        py_strings = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for stem, py_string in zip(batch['file_name'], py_strings):
            file_name = f'{stem}+{counter // len(dataset)}.py'
            with open(os.path.join(py_path, file_name), 'w') as f:
                f.write(py_string)
            counter += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='deepcad_test_mesh')
    parser.add_argument('--mode', type=str, default='pc')
    parser.add_argument('--checkpoint-path', type=str, default='maksimko123/cadrille')
    parser.add_argument('--py-path', type=str, default='./work_dirs/tmp_py')
    parser.add_argument('--input-source', type=str, default='mesh',
                        choices=['mesh', 'point_cloud', 'multi_view'])
    parser.add_argument('--mesh-ext', type=str, default='stl')
    parser.add_argument('--point-cloud-exts', type=str, default='ply,pcd,xyz,txt,npz,npy')
    parser.add_argument('--image-exts', type=str, default='png,jpg,jpeg,bmp')
    args = parser.parse_args()
    run(
        args.data_path,
        args.split,
        args.mode,
        args.checkpoint_path,
        args.py_path,
        args.input_source,
        tuple(ext.strip().lower() for ext in args.point_cloud_exts.split(',') if ext.strip()),
        tuple(ext.strip().lower() for ext in args.image_exts.split(',') if ext.strip()),
        args.mesh_ext.lower())
