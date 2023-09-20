import numpy as np 
import torch 
from torch import Tensor 

from typing import Union, Any

from functools import partial

def _is_numpy(x: Any) :
    return isinstance(x, np.ndarray);

def _is_torch(x: Any) :
    return torch.is_tensor(x);

def _assert_numpy_torch_2d(x: Union[np.ndarray, Tensor]) :
    assert x.ndim == 2, f"matrix must be 2D, got {x.ndim}D.";

def _assert_numpy_torch_square(x: Union[np.ndarray, Tensor]) :
    assert x.shape[0]==x.shape[1], \
        f"matrix must be square, got {x.shape}";

def _assert_real(x: np.ndarray) :
    _assert_numpy_torch_2d(x);
    if _is_numpy(x) :
        assert np.all(np.isreal(x)), f"matrix must be real.";
    else :
        assert torch.all(torch.isreal(x)), f"matrix must be real.";

def _assert_symmetric(x: np.ndarray) :
    _assert_numpy_torch_2d(x);
    if _is_numpy(x) :
        assert np.allclose(x, x.T), f"matrix must be symmetric";
    else :
        assert torch.allclose(x, x.T), f"matrix must be symmetric";



def get_eigs(
    var: Union[np.ndarray, Tensor],
) -> Union[np.ndarray, Tensor] :

    _assert_numpy_torch_2d(var);
    _assert_numpy_torch_square(var);
    _assert_real(var);
    _assert_symmetric(var);

    if _is_numpy(var) :
        eig_func = np.linalg.eig;
        abs_func = np.abs;
        sort_func = np.argsort;
    else :
        eig_func = torch.linalg.eig;
        abs_func = torch.abs;
        sort_func = partial(torch.argsort, descending=True);

    eig_vals, eig_vecs = eig_func(var);
    eig_vals = abs_func(eig_vals);
    eig_vecs = abs_func(eig_vecs);

    sorted_ids = sort_func(eig_vals)
    if _is_numpy(sorted_ids) :
        sorted_ids = sorted_ids[::-1];
    eig_vals = eig_vals[sorted_ids];
    eig_vecs = eig_vecs[:, sorted_ids];

    return eig_vecs, eig_vals;


def get_eig_vecs(
    var: Union[np.ndarray, Tensor],
    var_exp: float = 0.95,
) -> Union[np.ndarray, Tensor] :

    assert 0 < var_exp <= 1, \
        f"variance explanation factor must be in (0, 1], got {var_exp}";

    eig_vecs, eig_vals = get_eigs(var);

    eig_sum = eig_vals.sum();
    cum_var_exp = np.cumsum([(x / eig_sum) for x in eig_vals]);
    
    n_raw = eig_vecs.shape[1];
    n_keep = np.where(cum_var_exp<=var_exp)[0][-1] + 1;
    
    return eig_vecs[:, :n_keep], n_raw, n_keep;


def get_eig_projection(
    x: Union[np.ndarray, Tensor],
    eig_vecs: Union[np.ndarray, Tensor],
) -> np.ndarray :

    _assert_numpy_torch_2d(x);
    _assert_numpy_torch_2d(eig_vecs);
  
    is_numpy = _is_numpy(x) and _is_numpy(eig_vecs);
    is_torch = _is_torch(x) and _is_torch(eig_vecs); 
    assert is_numpy or is_torch, \
        f"Both x ({type(x)}) and eig_vecs ({type(eig_vecs)}) must be either np.ndarray or torch.Tensor";

    if x.shape[1] == eig_vecs.shape[1] :
        return x;

    return x @ eig_vecs;


def get_pca(
    x: Union[np.ndarray, Tensor],
    var_exp: float = 0.95,
) -> np.ndarray :

    _assert_numpy_torch_2d(x);
  
    cov_func = np.cov if _is_numpy(x) else torch.cov;
    var = cov_func(x.T);
    eig_vecs, _, _ = get_eig_vecs(var, var_exp);
    x_pca = get_eig_projection(x, eig_vecs);

    return x_pca;


def add_dummy_variances(
    var: Union[np.ndarray, Tensor],
) -> np.ndarray :

    _assert_numpy_torch_2d(var);
    if _is_numpy(var) :
        cumsum_func = np.cumsum;
        where_func = np.where;
        min_func = np.min;
        diag_func = np.diag;
    else :
        cumsum_func = partial(torch.cumsum, dim=0);
        where_func = torch.where;
        min_func = torch.min;
        diag_func = torch.diag;

    var_exp = 0.99;
    eig_vecs, eig_vals = get_eigs(var);

    eig_sum = eig_vals.sum();
    cum_var_exp = cumsum_func(eig_vals / eig_sum);
    
    n_raw = eig_vecs.shape[1];
    n_keep = where_func(cum_var_exp<=var_exp)[0][-1] + 1;
    eig_vals[n_keep:] = min_func(eig_vals[:n_keep]);

    var = eig_vecs @ diag_func(eig_vals) @ eig_vecs.T;    

    return var;    