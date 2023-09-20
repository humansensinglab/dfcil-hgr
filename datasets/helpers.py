import sys
import os, os.path as osp 

from typing import Sequence, List, Tuple, Optional, Any, IO, Union

import numpy as np

sys.path.append('..');

from utils.stdio import get_file_handle, load_pickle

def read_split_hgr_shrec_2017(
    root_dir: str, 
    split_filepath: str, 
    mode: str,
) -> List[Tuple[str]] :

    samples = [];
    print("Reading {} split from file = {}".format(mode, split_filepath));
    fhand = get_file_handle(split_filepath, 'r');

    for line in fhand :
        line = line.rstrip();
        file_path, label, size_seq, id_subject = line.split(',');
        samples.append((
            osp.join(root_dir, file_path), 
            int(label),
            int(size_seq),
            int(id_subject),
        ));
                   
    fhand.close();
    return samples;


def read_split_ego_gesture(
    root_dir: str, 
    split_filepath: str, 
    mode: str,
) -> List[Tuple[str]] :

    return read_split_hgr_shrec_2017(
                root_dir, 
                split_filepath,
                mode,
    );


def read_pts_hgr_shrec_2017(
    fpath: str, 
    rm_global_scale: bool = False,
) -> np.ndarray :

    fhand = get_file_handle(fpath, 'r');
    lines = ' '.join([x.strip() for x in fhand.readlines()]);
    pts = np.array(list(map(float, lines.split())), dtype=np.float32).reshape(-1, 22, 3);
    fhand.close();
    assert not np.any(np.isnan(pts));

    if rm_global_scale :
        return rm_global_scale_hgr_shrec_2017(pts);

    assert not np.any(np.isnan(pts));

    return pts;


def read_pts_mi_hgr_shrec_2017(
    fpath: str, 
    rm_global_scale: bool = False,
    is_inverted: bool = False,
) -> np.ndarray :

    if not is_inverted :
        return read_pts_hgr_shrec_2017(fpath, rm_global_scale);
        
    data = load_pickle(fpath);
    pts, label_feature, pred_feature = data['x'], data['label'], data['feature'];
    assert not np.any(np.isnan(pts));
    assert not np.any(np.isnan(label_feature));
    assert not np.any(np.isnan(pred_feature));

    if rm_global_scale :
        return rm_global_scale_hgr_shrec_2017(pts);

    assert not np.any(np.isnan(pts));

    return pts, label_feature, pred_feature;


def read_pts_ego_gesture(
    fpath: str, 
    rm_global_scale: bool = False,
) -> np.ndarray :

    fhand = get_file_handle(fpath, 'r');
    lines = ','.join([x.strip() for x in fhand.readlines()]);
    try :
        pts = np.array(list(map(float, lines.split(','))), dtype=np.float32).reshape(-1, 42, 3);
    except :
        pts = np.array(list(map(float, lines.split(','))), dtype=np.float32);
        print(pts);
        pts = pts.reshape(-1, 42, 3);
    fhand.close();
    assert not np.any(np.isnan(pts));

    if rm_global_scale :
        return rm_global_scale_ego_gesture(pts);

    assert not np.any(np.isnan(pts));

    return pts;

def read_pts_mi_ego_gesture(
    fpath: str, 
    rm_global_scale: bool = False,
    is_inverted: bool = False,
) -> np.ndarray :

    if not is_inverted :
        return read_pts_ego_gesture(fpath, rm_global_scale);
        
    data = load_pickle(fpath);
    pts, label_feature, pred_feature = data['x'], data['label'], data['feature'];
    assert not np.any(np.isnan(pts));
    assert not np.any(np.isnan(label_feature));
    assert not np.any(np.isnan(pred_feature));

    if rm_global_scale :
        return rm_global_scale_ego_gesture(pts);

    assert not np.any(np.isnan(pts));

    return pts, label_feature, pred_feature;


# def rm_global_scale_hgr_shrec_2017(pts: np.ndarray) -> np.ndarray :
#     pts_new = np.zeros_like(pts);
#     i_level = np.array(range(2, 19, 4), dtype=np.int32);

#     # wrist, palm
#     pts_new[:, 0] = pts[:, 0];
#     d = pts[:, 1] - pts[:, 0];
#     l = np.linalg.norm(d, axis=1, keepdims=True);
#     pts_new[:, 1] = pts[:, 0] + d / l;

#     # bases
#     l = l[:, None, :];
#     ds = pts[:, i_level] - pts[:, None, 1];
#     pts_new[:, i_level] = pts_new[:, None, 1] + ds / l;

#     # others
#     for k in range(1, 4) :
#         i_level += 1;
#         ds = pts[:, i_level] - pts[:, i_level-1];
#         pts_new[:, i_level] = pts_new[:, i_level-1] + ds / l;    

#     return pts_new;

# def rm_global_scale_hgr_shrec_2017(pts: np.ndarray) -> np.ndarray :
#     pts_new = np.zeros_like(pts);
#     i_level = np.array(range(2, 19, 4), dtype=np.int32);

#     # wrist, palm
#     d = pts[:, 1] - pts[:, 0];
#     l = np.linalg.norm(d, axis=1, keepdims=True);
#     pts_new[:, 0] = pts[:, 0] / l;
#     pts_new[:, 1] = pts_new[:, 0] + d / l;

#     # bases
#     l = l[:, None, :];
#     ds = pts[:, i_level] - pts[:, None, 1];
#     pts_new[:, i_level] = pts_new[:, None, 1] + ds / l;

#     # others
#     for k in range(1, 4) :
#         i_level += 1;
#         ds = pts[:, i_level] - pts[:, i_level-1];
#         pts_new[:, i_level] = pts_new[:, i_level-1] + ds / l;    

#     return pts_new;

def rm_global_scale_hgr_shrec_2017(pts: np.ndarray) -> np.ndarray :
    pts_new = np.zeros_like(pts);
    i_level = np.array(range(2, 19, 4), dtype=np.int32);

    # wrist, palm
    d = pts[:, 1] - pts[:, 0];
    l = np.mean(np.linalg.norm(d, axis=1, keepdims=True));
    pts_new[:, 0] = pts[:, 0] / l;
    pts_new[:, 1] = pts_new[:, 0] + d / l;

    # bases
    ds = pts[:, i_level] - pts[:, None, 1];
    pts_new[:, i_level] = pts_new[:, None, 1] + ds / l;

    # others
    for k in range(1, 4) :
        i_level += 1;
        ds = pts[:, i_level] - pts[:, i_level-1];
        pts_new[:, i_level] = pts_new[:, i_level-1] + ds / l;    

    # assert np.any(np.isnan(pts));
    # assert np.any(np.isnan(pts_new));

    return pts_new;


def rm_global_scale_ego_gesture(pts: np.ndarray) -> np.ndarray :
    def __rm_global_scale_single(pts: np.ndarray) -> np.ndarray :    
        pts_new = np.zeros_like(pts);
        i_level = np.array(range(1, 18, 4), dtype=np.int32);

        # wrist, index-base
        d = pts[:, 5] - pts[:, 0];
        l = np.mean(np.linalg.norm(d, axis=1, keepdims=True));
        pts_new[:, 0] = pts[:, 0] / l;
        # pts_new[:, 5] = pts_new[:, 5] + d / l;

        # bases
        ds = pts[:, i_level] - pts[:, None, 0];
        pts_new[:, i_level] = pts_new[:, None, 0] + ds / l;

        # others
        for k in range(1, 4) :
            i_level += 1;
            ds = pts[:, i_level] - pts[:, i_level-1];
            pts_new[:, i_level] = pts_new[:, i_level-1] + ds / l;    
        
        return pts_new;

    pts_left = pts[:, :21, :];
    pts_right = pts[:, 21:, :];
    if not np.allclose(pts_left, 0) :
        pts_left = __rm_global_scale_single(pts_left);
    if not np.allclose(pts_right, 0) :
        pts_right = __rm_global_scale_single(pts_right);
    
    pts = np.concatenate((pts_left, pts_right), axis=1);
    return pts;


def process_drop_list(drop_list: List[Union[int, str]]) :
    if drop_list is None or len(drop_list) == 0 :
        return [];
    
    for i in range(len(drop_list)) :
        drop_list[i] = int(drop_list[i]);
    
    return drop_list;

def drop_classes_hgr_shrec_2017(
    file_list: List[Tuple[str]],
    drop_list: List[int],
) -> List[Tuple[str]] :

    if drop_list is None or len(drop_list) == 0 :
        return file_list;

    file_list_new = [];
    for sample in file_list :
        _, label, _, _ = sample;
        if label in drop_list :
            continue;
        
        file_list_new.append(sample);
    
    return file_list_new;


def drop_classes_ego_gesture(
    file_list: List[Tuple[str]],
    drop_list: List[int],
) -> List[Tuple[str]] :

    return drop_classes_hgr_shrec_2017(
            file_list,
            drop_list,
    );


def keep_classes_hgr_shrec_2017(
    file_list: List[Tuple[str]],
    keep_class_l: List[int],
) -> List[Tuple[str]] :

    assert len(keep_class_l)>0, \
            f"There must be at least one class to keep.";

    keep_class_l = set(keep_class_l);
    file_list_new = [];
    for sample in file_list :
        _, label, _, _ = sample;
        if label not in keep_class_l :
            continue;
        file_list_new.append(sample);
    
    return file_list_new;


def keep_classes_ego_gesture(
    file_list: List[Tuple[str]],
    keep_class_l: List[int],
) -> List[Tuple[str]] :

    return keep_classes_hgr_shrec_2017(
            file_list,
            keep_class_l,
    );    


def get_file_list(
    dataset_name: str,
    root_dir: str,
    split_filepath: str,
    mode: str,
    keep_class_l: Optional[ List[int] ] = None,
) -> List[Any] :

    file_list = globals()['read_split_' + dataset_name](root_dir, split_filepath, mode);
    if keep_class_l is not None :
        file_list = globals()['keep_classes_' + dataset_name](file_list, keep_class_l);        
    return file_list;


