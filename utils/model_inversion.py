import sys
import os, os.path as osp
import shutil
import time
from tqdm import tqdm

import numpy as np 
import random
import torch

from typing import IO

from .svd import get_eig_vecs
from .classifier import *
from .stdio import *


def _normalize(x: np.ndarray) -> np.ndarray :
    return x / np.linalg.norm(x);


def _get_unit_vectors(
    n: int, 
    d: int, 
) -> np.ndarray :

    vecs = np.random.rand(n, d);
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True);
    return vecs;


def _assert_unique(x: np.ndarray) -> None :
    from scipy.spatial import distance_matrix
    d = distance_matrix(x, x);
    assert d.size - np.count_nonzero(d) == x.shape[0];    


def get_circular_samples(
    dim: int, 
    order: int = 3,
    dtype: str = 'float32',
) -> np.ndarray :

    assert 1 <= order <= 3, \
        f"Currently only support upto 3rd order sample generation, got order {order}.";

    normalize = lambda x : x / np.linalg.norm(x);

    samples_1 = np.eye(dim);

    if order == 1 :
        samples = np.concatenate(
            (
                samples_1, 
                -samples_1,
            ), axis=0
        );        

        return samples;

    samples_2 = [];
    for i in range(dim-1) :
        for j in range(i+1, dim) :
            
            samples_2.append(
                normalize( samples_1[:, i] + samples_1[:, j] )
            );
            samples_2.append(
                normalize( samples_1[:, i] - samples_1[:, j] )
            );            

    samples_2 = np.stack(samples_2, axis=1);

    if order == 2 :
        samples = np.concatenate(
            (
                samples_1, 
                samples_2, 
                -samples_1,
                -samples_2,
            ), axis=1
        );        

        return samples;

    samples_3 = [];
    for i in range(dim-2) :
        for j in range(i+1, dim-1) :
            for k in range(j+1, dim) :
                
                samples_3.extend([
                    normalize( samples_1[:, i] + samples_1[:, j] + samples_1[:, k] ), 
                    normalize( samples_1[:, i] + samples_1[:, j] - samples_1[:, k] ), 
                    normalize( samples_1[:, i] - samples_1[:, j] + samples_1[:, k] ), 
                    normalize( -samples_1[:, i] + samples_1[:, j] + samples_1[:, k] ), 
                ]);



    samples_3 = np.stack(samples_3, axis=1);

    samples = np.concatenate(
        (
            samples_1, 
            samples_2, 
            samples_3,
            -samples_1,
            -samples_2,
            -samples_3
        ), axis=1
    );
    
    samples = samples.T;

    return samples;


def _clamp_sample_size(
    xs: np.ndarray,
    max_samples: int,
) -> np.ndarray :

    n_samples = xs.shape[0];
    if n_samples <= max_samples :
        return xs;

    i_select = np.random.RandomState(seed=0).permutation(n_samples)[:max_samples];
    xs = xs[i_select];
    return xs;


def get_mixture_of_samples(
    samples: np.ndarray, 
    dim: int,
    n_samples: int,
    max_samples: int,
    order: int = 3,
    dtype: str = 'float32',
) -> np.ndarray :

    assert 1 <= order <= 4, \
        f"Currently only support upto 4th order sample generation, got order {order}.";

    assert samples.ndim == 2, f"samples must be 2D array but got {samples.shape}";
    assert n_samples == samples.shape[0] and dim == samples.shape[1], \
        f"samples must be of shape ({n_samples}, {dim}), but got {samples.shape}";

    samples_1 = np.copy(samples);
    for i in range(n_samples) :
        samples_1[i] = _normalize(samples_1[i]);

    samples = np.concatenate(
        (
            samples_1, 
            -samples_1,
        ), axis=0
    );        

    n_samples_1 = samples.shape[0];
    if n_samples_1 >= max_samples :
        samples = _clamp_sample_size(samples, max_samples);
        _assert_unique(samples);
        return samples; 
   
    if order == 1 :
        _assert_unique(samples);
        return samples;

    samples_2 = [];
    for i in range(n_samples-1) :
        for j in range(i+1, n_samples) :

            samples_2.append(
                _normalize( samples_1[i] + samples_1[j] )
            );
            samples_2.append(
                _normalize( samples_1[i] - samples_1[j] )
            );            

    samples_2 = np.stack(samples_2, axis=0);

    samples_2 = np.concatenate(
        (
            samples_2, 
            -samples_2,
        ), axis=0
    );        

    n_samples_2 = samples_2.shape[0];
    max_samples = max_samples - n_samples_1;
    if n_samples_2 >= max_samples :
        samples_2 = _clamp_sample_size(samples_2, max_samples);
        samples = np.concatenate(
            (
                samples, 
                samples_2,
            ), axis=0
        );      

        _assert_unique(samples);
        return samples; 
    else :
        samples = np.concatenate(
            (
                samples, 
                samples_2,
            ), axis=0
        );              
   
    if order == 2 :
        _assert_unique(samples);
        return samples;

    samples_3 = [];
    for i in range(n_samples-2) :
        for j in range(i+1, n_samples-1) :
            for k in range(j+1, n_samples) :

                samples_3.extend([
                    _normalize( samples_1[i] + samples_1[j] + samples_1[k] ), 
                    _normalize( samples_1[i] + samples_1[j] - samples_1[k] ), 
                    _normalize( samples_1[i] - samples_1[j] + samples_1[k] ), 
                    _normalize( -samples_1[i] + samples_1[j] + samples_1[k] ), 
                ]);



    samples_3 = np.stack(samples_3, axis=0);

    samples_3 = np.concatenate(
        (
            samples_3, 
            -samples_3,
        ), axis=0
    );       

    n_samples_3 = samples_3.shape[0];
    max_samples = max_samples - n_samples_2;
    if n_samples_3 >= max_samples :
        samples_3 = _clamp_sample_size(samples_3, max_samples);
        samples = np.concatenate(
            (
                samples, 
                samples_3,
            ), axis=0
        );      

        _assert_unique(samples);
        return samples; 
    
    else :
        samples = np.concatenate(
            (
                samples, 
                samples_3,
            ), axis=0
        );        
   
    if order == 3 :
        _assert_unique(samples);
        return samples;

    samples_4 = [];
    for i in range(n_samples-3) :
        for j in range(i+1, n_samples-2) :
            for k in range(j+1, n_samples-1) :
                for q in range(k+1, n_samples) :
                    s1, s2, s3, s4 = samples_1[i], samples_1[j], samples_1[k], samples_1[q];
                    s_pos = s1 + s2 + s3 + s4;
                    samples_4.extend([
                        _normalize( s_pos ), 
                        _normalize( -s_pos ), 
                        _normalize( s_pos - 2 * s1 ), 
                        _normalize( s_pos - 2 * s2 ), 
                        _normalize( s_pos - 2 * s3 ), 
                        _normalize( s_pos - 2 * s4 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s2 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s3 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s4 ), 
                        _normalize( s_pos - 2 * s2 - 2 * s3 ), 
                        _normalize( s_pos - 2 * s2 - 2 * s4 ), 
                        _normalize( s_pos - 2 * s3 - 2 * s4 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s2 - 2 * s3 ), 
                        _normalize( s_pos - 2 * s1 - 2 * s3 - 2 * s4 ), 
                        _normalize( s_pos - 2 * s2 - 2 * s3 - 2 * s4 ), 
                    ]);



    samples_4 = np.stack(samples_4, axis=0);

    n_samples_4 = samples_4.shape[0];
    max_samples = max_samples - n_samples_3;
    if n_samples_4 >= max_samples :
        samples_4 = _clamp_sample_size(samples_4, max_samples);
        samples = np.concatenate(
            (
                samples, 
                samples_4,
            ), axis=0
        );      

        _assert_unique(samples);
        return samples; 
    else :
        samples = np.concatenate(
            (
                samples, 
                samples_4,
            ), axis=0
        );              
   
    _assert_unique(samples);
    return samples;


def get_elliptical_samples(
    proto_class_var: np.ndarray, 
    var_exp: float,
    order: int,
    max_samples: int,
) -> np.ndarray :

    eig_vecs, _, _ = get_eig_vecs(proto_class_var, var_exp); 

    d, n = eig_vecs.shape;
    principal_axes = get_mixture_of_samples(eig_vecs.T, d, n, max_samples, order);
    return principal_axes;


def get_inverted_sample_single_w_svm(
    model, clf, 
    proto_mean, p_ax, class_id, 
    params,
    x_proto_mean,
) :
    def __get_norm_dist(pred, target) :
        norm_ = torch.norm(target).item();
        d = torch.dist(pred, target) / norm_;   
        return d;

    def __classify(model, clf, feature) :
        feature = feature.data.cpu().numpy();
        return predict_classifier(clf, feature).item();

    def __is_same_class(model, clf, feature, class_id) :
        return __classify(model, clf, feature) == class_id;

    def __get_expansion(x1, x2) :
        return torch.norm(x2).item() / torch.norm(x1).item();

    @torch.no_grad()
    def __init_input(model, dtype, device) :
        x = torch.zeros((1, *model.get_input_shape())).to(dtype).to(device);
        x.requires_grad_(True);
        return x;

    lr_f, lr, momentum, tol, tol_ub, max_iter_f, max_iter = params;

    model.eval();

    assert __is_same_class(model, clf, proto_mean, class_id);

    shift_ = lr_f * p_ax;
    feature = proto_mean.clone();

    for i in range(max_iter_f) :
        feature.add_(shift_);
        if not __is_same_class(model, clf, feature, class_id) :
            d = feature.data - proto_mean.data;
            feature = proto_mean.data + 0.8 * d; 
            assert __is_same_class(model, clf, feature, class_id);
            break;            
    
    target = feature.clone();

    dtype = torch.get_default_dtype();
    device = target.device; 
    
    x = x_proto_mean.clone().detach();
    x.requires_grad_(True);    
    
    m_grad = torch.zeros_like(x);
    for i in range(max_iter) :
        feat = model.forward_feature(x);

        d = __get_norm_dist(feat, target);
        if d.item() < tol :
            break;
      
        grad_x = torch.autograd.grad(d, x)[0];
        m_grad = momentum * m_grad + grad_x; 
        x = x - lr * m_grad;

    feat = model.forward_feature(x);
    feat_np = feat.data.cpu().numpy();
    pred_class_id = predict_classifier(clf, feat_np)[0];

    d = d.item();

    if (pred_class_id != class_id) :
        return None, None, None, d;    
    return x, target, feat, d;


def get_inverted_sample_random_single_w_svm(
    model, clf, 
    proto_mean, p_ax, class_id, 
    params, 
    x_proto_mean,
) :
    def __get_norm_dist(pred, target) :
        norm_ = torch.norm(target).item();
        d = torch.dist(pred, target) / norm_;  
        return d;

    def __classify(model, clf, feature) :
        feature = feature.data.cpu().numpy();
        return predict_classifier(clf, feature).item();

    def __is_same_class(model, clf, feature, class_id) :
        return __classify(model, clf, feature) == class_id;

    def __get_expansion(x1, x2) :
        return torch.norm(x2).item() / torch.norm(x1).item();

    @torch.no_grad()
    def __init_input(model, dtype, device) :
        x = torch.zeros((1, *model.get_input_shape())).to(dtype).to(device);
        x.requires_grad_(True);
        return x;

    lr_f, lr, momentum, tol, tol_ub, max_iter_f, max_iter = params;

    model.eval();

    assert __is_same_class(model, clf, proto_mean, class_id);

    shift_ = lr_f * p_ax;
    feature = proto_mean.clone();

    for i in range(max_iter_f) :
        feature.add_(shift_);
        if not __is_same_class(model, clf, feature, class_id) :
            d = feature.data - proto_mean.data;
            rand_shift = np.random.uniform(0, 1);
            feature = proto_mean.data + rand_shift * d;
            assert __is_same_class(model, clf, feature, class_id);
            break;
    
    target = feature.clone();

    dtype = torch.get_default_dtype();
    device = target.device; 
    
    x = x_proto_mean.clone().detach();
    x.requires_grad_(True);    
    
    m_grad = torch.zeros_like(x);
    for i in range(max_iter) :
        feat = model.forward_feature(x);

        d = __get_norm_dist(feat, target);
        if d.item() < tol :
            break;
       
        grad_x = torch.autograd.grad(d, x)[0];
        m_grad = momentum * m_grad + grad_x; 
        x = x - lr * m_grad;

    feat = model.forward_feature(x);
    feat_np = feat.data.cpu().numpy();
    pred_class_id = predict_classifier(clf, feat_np)[0];

    d = d.item();
    if (pred_class_id != class_id) :
        return None, None, None, d;    
    return x, target, feat, d;


def get_inverted_proto_mean_single(
    model, clf, 
    proto_mean, class_id, 
    params) :
    def __get_norm_dist(pred, target) :
        norm_ = torch.norm(target).item();
        d = torch.dist(pred, target) / norm_;     
        return d;

    def __classify(model, clf, feature) :
        feature = feature.data.cpu().numpy();
        return predict_classifier(clf, feature).item();

    def __is_same_class(model, clf, feature, class_id) :
        return __classify(model, clf, feature) == class_id;

    @torch.no_grad()
    def __init_input(model, dtype, device) :
        x = torch.zeros((1, *model.get_input_shape())).to(dtype).to(device);
        x.requires_grad_(True);
        return x;

    lr, momentum, tol, tol_ub = params;

    model.eval();

    assert __is_same_class(model, clf, proto_mean, class_id);

    target = proto_mean.clone();

    dtype = torch.get_default_dtype();
    device = target.device; 
    
    x = __init_input(model, dtype, device);
    
    m_grad = torch.zeros_like(x);
    for i in range(10000) :
        feat = model.forward_feature(x);

        d = __get_norm_dist(feat, target);
        if d.item() < tol :
            break;       
        
        grad_x = torch.autograd.grad(d, x)[0];
        m_grad = grad_x; 
        x = x - lr * m_grad;

    feat = model.forward_feature(x);
    feat_np = feat.data.cpu().numpy();
    pred_class_id = predict_classifier(clf, feat_np)[0];

    assert (pred_class_id == class_id), f"Class id = {class_id}"; 

    if (d > tol_ub) : 
        print(f"Class id, d = {class_id}, {d:.4f}"); 

    return x;    


def get_mixture_of_samples_svm(
    samples: np.ndarray, 
    dim: int,
    n_samples: int,
    order: int = 3,
    dtype: str = 'float32',
) -> np.ndarray :

    assert 2 <= order <= 3, \
        f"Currently only support upto 3rd order sample generation, got order {order}.";

    assert samples.ndim == 2, f"samples must be 2D array but got {samples.shape}";
    assert n_samples == samples.shape[0] and dim == samples.shape[1], \
        f"samples must be of shape ({n_samples}, {dim}), but got {samples.shape}";

    samples_1 = np.copy(samples);

    samples_2 = [];
    for i in range(n_samples-1) :
        for j in range(i+1, n_samples) :
            samples_2.append(
                samples_1[i] + samples_1[j]
            );

    samples_2 = np.stack(samples_2, axis=0);

    if order == 2 :
        return samples_2;

    samples_3 = [];
    for i in range(n_samples-2) :
        for j in range(i+1, n_samples-1) :
            for k in range(j+1, n_samples) :
                samples_3.append(
                    samples_1[i] + samples_1[j] + samples_1[k] 
                );

    samples_3 = np.stack(samples_3, axis=0);
    samples = np.concatenate(
        (
            samples_2, 
            samples_3,
        ), axis=0
    );
    
    return samples;


def get_inverted_sample_svm_single(
    model, clf, 
    sv, class_id, 
    params, 
    x_proto_mean,
) :
    def __get_norm_dist(pred, target) :
        norm_ = torch.norm(target).item();
        d = torch.dist(pred, target) / norm_;
        return d;

    def __classify(model, x, clf=None) :
        if clf is None :
            return model(x).argmax(1).item();
        
        feat = model.forward_feature(x);
        feat = feat.data.cpu().numpy();
        return predict_classifier(clf, feat);

    @torch.no_grad()
    def __init_input(model, dtype, device) :
        x = torch.zeros((1, *model.get_input_shape())).to(dtype).to(device);
        x.requires_grad_(True);
        return x;


    lr, momentum, tol, tol_ub, max_iter = params;

    model.eval();

    target = sv.clone();

    dtype = torch.get_default_dtype();
    device = target.device; 
    
    x = x_proto_mean.clone().detach();
    x.requires_grad_(True);
    
    m_grad = torch.zeros_like(x);

    for i in range(max_iter) : 
        feat = model.forward_feature(x);

        d = __get_norm_dist(feat, target);
        if d.item() < tol :
            break;

        grad_x = torch.autograd.grad(d, x)[0];
        m_grad = momentum * m_grad + grad_x;
        x = x - lr * m_grad;

    d = d.item();
    pred_class_id = __classify(model, x, clf)[0];
    if (pred_class_id != class_id) :
        return None, None, None;    
    return x, target, feat;


def get_inverted_samples_svm_mixtures(model, clf, svs_mixture, class_id, mult_, lr, tol) :
    def __get_norm_dist(pred, target) :
        norm_ = torch.norm(target).item();
        d = torch.sum(torch.pow((pred - target), 2)) / norm_;       
        return d;

    def __classify(model, x, clf=None) :
        if clf is None :
            return model(x).argmax(1).item();
        
        feat = model.forward_feature(x);
        feat = feat.data.cpu().numpy();
        return predict_classifier(clf, feat);


    def __get_sv_from_mixture(clf, sv) :
        def get_directed_sv_from_mixture(clf, feature, shift_, max_iter=100) :
            st_begin = predict_classifier(clf, feature).item();
            feature_pos = np.copy(feature);
            feature_neg = np.copy(feature);
            shift_pos = shift_;
            shift_neg = -shift_;
            for _ in range(max_iter) :
                feature_pos = feature_pos + shift_pos;
                feature_neg = feature_neg + shift_neg;

                st_end_pos = predict_classifier(clf, feature_pos).item();
                st_end_neg = predict_classifier(clf, feature_neg).item();
                
                if st_end_pos != st_begin :
                    feature = feature - shift_pos; 
                    break;
                
                if st_end_neg != st_begin :
                    feature = feature - shift_neg; 
                    break;
            
            if (st_begin == st_end_pos) and (st_begin == st_end_neg) : 
                return None;
            
            return feature;

        shift_ = mult_ * sv;
        return get_directed_sv_from_mixture(clf, sv, shift_);
        

    sv_l = [];
    for k in range(svs_mixture.shape[0]) :
        sv = svs_mixture[k:k+1];
        sv = __get_sv_from_mixture(clf, sv);
        if sv is None :
            continue;
        sv_l.append(sv);
    
    if len(sv_l) == 0 :
        print(f"No mixture sample is converted to SV for class = {class_id}");
        return None;

    svs_mixture = np.concatenate(sv_l, axis=0);
    return svs_mixture;    


def save_inverted_samples_svm_parallel(info) :
    out_dir, model, clf, proto_var, proto_mean, \
        class_id, mult_, lr, tol, \
        max_samples_per_class = info;

    dtype = torch.get_default_dtype();
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; 

    svs = get_svs_w_classes(clf, class_id);   
    svs_mixture = get_mixture_of_samples_svm(
        svs, svs.shape[1], svs.shape[0], order=2, 
    );
    n_mixture = max_samples_per_class - svs.shape[0];
    if svs_mixture.shape[0] < n_mixture :
        svs_mixture = get_mixture_of_samples_svm(
            svs, svs.shape[1], svs.shape[0], order=3, 
        );        

    if n_mixture > 0 :
        i_select = np.random.RandomState(seed=0).permutation(svs_mixture.shape[0])[:n_mixture];
        svs_mixture = svs_mixture[i_select];

    svs_mixture = get_inverted_samples_svm_mixtures(
        model, clf, svs_mixture, class_id, mult_, lr, tol
    );
    
    svs = np.concatenate((svs, svs_mixture), axis=0);

    svs_t = torch.from_numpy(svs).to(dtype).to(device);
    count = 0;
    n_samples = svs_t.size(0);
    for i in range(n_samples) :
        print(f"[{i+1} / {n_samples}] Class = {class_id}", flush=True);

        sv_i = svs_t[i:i+1];
        x = get_inverted_sample_svm_single(model, clf, sv_i, class_id, mult_, lr, tol);
        if x is None :
            continue;
        assert x.size(0) == 1;
        x = x.view(x.size(1), -1);
        x = x.data.cpu().numpy();
        x = x.astype(np.float32);

        fname = str(count).zfill(6) + '.txt';
        np.savetxt(osp.join(out_dir, fname), x, fmt='%.6f', delimiter=' ');
        count += 1;

    print(f"Class ({class_id}) => Samples saved = {count}");  


def save_inverted_samples_proto_svm_parallel_v2(
    out_dir, 
    model, clf, 
    proto_mean, proto_var,
    mi_params,
) :

    def __invert_single_sample(info) :
        i, fpath, model, clf, mi_params = info;
        data = load_pickle(fpath);
        sample_type = data['type'];
        out_dir = data['out_dir'];
        class_id = data['class_id'];
        x_proto_mean = data['x_proto_mean'];

        dtype = torch.get_default_dtype();
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; 

        x_proto_mean = torch.from_numpy(x_proto_mean).to(dtype).to(device);
        if sample_type == 'sv' :
            sv = data['sv'];
            sv_t = torch.from_numpy(sv).to(dtype).to(device);
            if sv_t.ndim == 1 :
                sv_t.unsqueeze_(0);
            sv_t.requires_grad_(False);
            x, target, feat = \
                get_inverted_sample_svm_single(
                    model, clf, sv_t, class_id, 
                    mi_params,
                    x_proto_mean);            

        elif sample_type == 'proto' :
            proto_mean = data['proto_mean'];
            p_ax = data['p_ax'];

            proto_mean_t = torch.from_numpy(proto_mean).to(dtype).to(device);
            p_ax_t = torch.from_numpy(p_ax).to(dtype).to(device);

            if proto_mean_t.ndim == 1 :
                proto_mean_t.unsqueeze_(0);
            if p_ax_t.ndim == 1 :
                p_ax_t.unsqueeze_(0);
            p_ax_t.requires_grad_(False);
            x, target, feat, d_perc = \
                get_inverted_sample_single_w_svm(
                    model, clf, 
                    proto_mean_t, p_ax_t, class_id, 
                    mi_params, 
                    x_proto_mean,
                );
        else :
            raise NotImplementedError;

        if x is None :
            print(f"{i} => {class_id} (failure)", flush=True);
            return;

        print(f"{i} => {class_id}", flush=True);
        assert x.size(0) == 1;
        x = x.view(x.size(1), -1);
        x = x.data.cpu().numpy();
        x = x.astype(np.float32);
        target = target.squeeze().data.cpu().numpy();
        target = target.astype(np.float32);
        feat = feat.squeeze().data.cpu().numpy();
        feat = feat.astype(np.float32);        

        save_dict = {
            'x': x,
            'label': target,
            'feature': feat,
        };
        fname = str(i).zfill(6) + '.pkl';
        save_pickle(osp.join(out_dir, fname), save_dict);


    import concurrent.futures as futures
    infos = [];
    tmp_out_dir = osp.join(out_dir, 'tmp');
    mkdir_rm_if_exists(tmp_out_dir);

    save_proto_svm_mi_infos(
        tmp_out_dir,
        out_dir, 
        model, clf,
        proto_mean, proto_var,
        mi_params,
        infos,
    );

    print(f"Trying to invert {len(infos)} samples ...");
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(__invert_single_sample, infos);  

    shutil.rmtree(tmp_out_dir);  


def save_inverted_samples_random_svm(
    out_dir, 
    model, clf, 
    proto_mean, proto_var,
    mi_params,
) :

    def __invert_single_sample(info) :
        i, fpath, model, clf, mi_params = info;
        data = load_pickle(fpath);
        sample_type = data['type'];
        out_dir = data['out_dir'];
        class_id = data['class_id'];
        x_proto_mean = data['x_proto_mean'];

        dtype = torch.get_default_dtype();
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; 

        x_proto_mean = torch.from_numpy(x_proto_mean).to(dtype).to(device);
        if sample_type == 'random_svm' :
            proto_mean = data['proto_mean'];
            p_ax = data['p_ax'];

            proto_mean_t = torch.from_numpy(proto_mean).to(dtype).to(device);
            p_ax_t = torch.from_numpy(p_ax).to(dtype).to(device);

            if proto_mean_t.ndim == 1 :
                proto_mean_t.unsqueeze_(0);
            if p_ax_t.ndim == 1 :
                p_ax_t.unsqueeze_(0);
            p_ax_t.requires_grad_(False);
            x, target, feat, d_perc = \
                get_inverted_sample_random_single_w_svm(
                    model, clf, proto_mean_t, p_ax_t, class_id, 
                    mi_params, 
                    x_proto_mean,
                );

        else :
            raise NotImplementedError;

        if x is None :
            print(f"{i} => {class_id} (failure)", flush=True);
            return;

        print(f"{i} => {class_id}", flush=True);
        assert x.size(0) == 1;
        x = x.view(x.size(1), -1);
        x = x.data.cpu().numpy();
        x = x.astype(np.float32);
        target = target.squeeze().data.cpu().numpy();
        target = target.astype(np.float32);
        feat = feat.squeeze().data.cpu().numpy();
        feat = feat.astype(np.float32);        

        save_dict = {
            'x': x,
            'label': target,
            'feature': feat,
        };
        fname = str(i).zfill(6) + '.pkl';
        save_pickle(osp.join(out_dir, fname), save_dict);


    import concurrent.futures as futures
    infos = [];
    tmp_out_dir = osp.join(out_dir, 'tmp');
    mkdir_rm_if_exists(tmp_out_dir);

    save_random_svm_mi_infos(
        tmp_out_dir,
        out_dir, 
        model, clf,
        proto_mean, proto_var,
        mi_params,
        infos,
    );

    print(f"Trying to invert {len(infos)} samples ...");
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(__invert_single_sample, infos);  

    shutil.rmtree(tmp_out_dir);      


def save_proto_svm_mi_infos(
    tmp_out_dir,
    out_dir, 
    model, clf, 
    proto_mean, proto_var,
    mi_params,
    infos,
) :

    add_svm, add_proto = False, False;
    inv_type = mi_params.inv_type;
    if inv_type == 'proto-svm' :
        add_svm, add_proto = True, True;
    elif inv_type == 'svm' :
        add_svm = True;
    elif inv_type == 'proto' :
        add_proto = True;
    else :
        raise NotImplementedError;

    x_proto_mean = {};
    dtype = torch.get_default_dtype();
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; 

    pm_params = [
        mi_params.lr.backward,
        mi_params.momentum,
        mi_params.tol.sv,
        mi_params.tol_ub.sv,
    ]

    for class_id in proto_mean :
        print(f"Inverting proto mean for class = {class_id}");

        proto_mean_c = proto_mean[class_id];
        proto_var_c = proto_var[class_id];

        proto_mean_t = torch.from_numpy(proto_mean_c).to(dtype).to(device);
        x = get_inverted_proto_mean_single(model, clf, proto_mean_t, class_id, pm_params);
        x = x.data.cpu().numpy();
        x = x.astype(np.float32);        
        x_proto_mean[class_id] = x;

    print(f"Inversion of proto means are done.");

    sv_params = [
        mi_params.lr.backward,
        mi_params.momentum,
        mi_params.tol.sv,
        mi_params.tol_ub.sv,
        mi_params.max_iter.backward.sv,        
    ];

    proto_params = [
        mi_params.lr.forward,
        mi_params.lr.backward,
        mi_params.momentum,
        mi_params.tol.proto,
        mi_params.tol_ub.proto,
        mi_params.max_iter.forward,
        mi_params.max_iter.backward.proto,
    ];  

    proto_order = mi_params.order;
    proto_var_exp = mi_params.var_exp;  

    n_samples, n_prev_samples = 0, 0;
    n_sv_samples, n_proto_samples = 0, 0;
    for class_id in proto_mean :
        print(f"Saving info for class = {class_id}");
        out_dir_c = osp.join(out_dir, 'class_' + str(class_id));
        mkdir_rm_if_exists(out_dir_c);

        proto_mean_c = proto_mean[class_id];
        proto_var_c = proto_var[class_id];

        if add_svm :
            svs = get_svs_w_classes(clf, class_id);  
            for i in range(svs.shape[0]) :
                data = {
                    'type': 'sv', 
                    'out_dir': out_dir_c,
                    'class_id': class_id,
                    'sv': svs[i],
                    'x_proto_mean': x_proto_mean[class_id],
                };

                fpath = osp.join(tmp_out_dir, str(n_samples).zfill(6) + '.pkl');
                save_pickle(fpath, data);
                infos.append( (
                    n_samples,
                    fpath,
                    model, 
                    clf,
                    sv_params,
                ) );

                n_samples += 1;

            n_sv_samples = n_samples - n_prev_samples;
            print(f"Class ({class_id}) => SV samples saved = {n_sv_samples}");

        if add_proto :
            principal_axes = \
                get_elliptical_samples(
                    proto_var_c, 
                    order=proto_order, 
                    var_exp=proto_var_exp,
                    max_samples=mi_params.max_samples_per_class - n_sv_samples,
            );

            for i in range(principal_axes.shape[0]) :
                data = {
                    'type': 'proto', 
                    'out_dir': out_dir_c,
                    'class_id': class_id,
                    'proto_mean': proto_mean_c,
                    'p_ax': principal_axes[i],
                    'x_proto_mean': x_proto_mean[class_id],        
                };             

                fpath = osp.join(tmp_out_dir, str(n_samples).zfill(6) + '.pkl');
                save_pickle(fpath, data);
                infos.append( (
                    n_samples,
                    fpath,
                    model, 
                    clf,
                    proto_params,
                ) );

                n_samples += 1;

            n_proto_samples = n_samples - n_prev_samples - n_sv_samples;
            print(f"Class ({class_id}) => Proto samples saved = {n_proto_samples}");
            print();

        n_prev_samples = n_samples;



def save_random_svm_mi_infos(
    tmp_out_dir,
    out_dir, 
    model, clf, 
    proto_mean, proto_var,
    mi_params,
    infos,
) :

    x_proto_mean = {};
    dtype = torch.get_default_dtype();
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'; 

    pm_params = [
        mi_params.lr.backward,
        mi_params.momentum,
        mi_params.tol.sv,
        mi_params.tol_ub.sv,
    ]

    for class_id in proto_mean :
        print(f"Inverting proto mean for class = {class_id}");
        sys.stdout.flush();

        proto_mean_c = proto_mean[class_id];
        proto_var_c = proto_var[class_id];

        proto_mean_t = torch.from_numpy(proto_mean_c).to(dtype).to(device);
        x = get_inverted_proto_mean_single(model, clf, proto_mean_t, class_id, pm_params);
        x = x.data.cpu().numpy();
        x = x.astype(np.float32);        
        x_proto_mean[class_id] = x;

    print(f"Inversion of proto means are done.");

    proto_params = [
        mi_params.lr.forward,
        mi_params.lr.backward,
        mi_params.momentum,
        mi_params.tol.proto,
        mi_params.tol_ub.proto,
        mi_params.max_iter.forward,
        mi_params.max_iter.backward.proto,
    ]; 

    n_samples, n_prev_samples = 0, 0;
    for class_id in proto_mean :
        print(f"Saving info for class = {class_id}");
        out_dir_c = osp.join(out_dir, 'class_' + str(class_id));
        mkdir_rm_if_exists(out_dir_c);

        proto_mean_c = proto_mean[class_id];
        proto_var_c = proto_var[class_id];

        dim = proto_mean_c.shape[1];
        normals = _get_unit_vectors(mi_params.max_samples_per_class, dim);
        for i in range(normals.shape[0]) :
            data = {
                'type': 'random_svm', 
                'out_dir': out_dir_c,
                'class_id': class_id,
                'proto_mean': proto_mean_c,
                'p_ax': normals[i],
                'x_proto_mean': x_proto_mean[class_id],        
            };             

            fpath = osp.join(tmp_out_dir, str(n_samples).zfill(6) + '.pkl');
            save_pickle(fpath, data);
            infos.append( (
                n_samples,
                fpath,
                model, 
                clf,
                proto_params,
            ) );

            n_samples += 1;

        n_random_samples = n_samples - n_prev_samples;
        print(f"Class ({class_id}) => Random samples saved = {n_random_samples}");
        print();

        n_prev_samples = n_samples;


def write_train_val_splits(
    sample_dir: str, 
    crop_len_dir: int, 
    f_train: IO, 
    f_val: IO,
    class_id: int,
    seq_len: int,
    val_perc: float = 0.2,
) -> None :

    file_list = sorted(os.listdir(sample_dir));
    random.shuffle(file_list);

    n_val = int(round(len(file_list) * val_perc));

    sample_subdir = sample_dir[crop_len_dir:];
    class_id = str(class_id);
    seq_len = str(seq_len);
    id_subject = str(-1);
    for fname in sorted(file_list[:n_val]) :
        f_val.write(
            osp.join(sample_subdir, fname) + ',' + \
            class_id + ',' + \
            seq_len + ',' + \
            id_subject + '\n'
        );

    for fname in sorted(file_list[n_val:]) :
        f_train.write(
            osp.join(sample_subdir, fname) + ',' + \
            class_id + ',' + \
            seq_len + ',' + \
            id_subject + '\n'
        );  

    n_train = len(file_list) - n_val;
    print(f"Class ({class_id}) => Samples saved (Train / Val)= ({n_train} / {n_val})");