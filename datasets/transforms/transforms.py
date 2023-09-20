from typing import List, Tuple, Sequence, Union, Optional, Any

from easydict import EasyDict as edict

import math
import numpy as np
import random
from numbers import Number

import torch

def _pair(
    x : Union[Sequence[Number], Number],
    negate: bool = False,
) -> Sequence[Number] :

    if isinstance(x, Number) :
        if negate : 
            x = abs(x);
            return [-x, x];
        return [x, x];
    return x;

def _deg2rad(
    x : Sequence[Number],
) -> Sequence[Number] :

    return [d * math.pi / 180 for d in x];

def _is_numpy(x: Any) -> bool :
    return isinstance(x, np.ndarray);

def _is_torch(x: Any) -> bool :
    return torch.is_tensor(x);


class Compose(object) :
    """Composes several transforms together for multiple input arrays at once.
    If you need a detailed documentation, feel free to read from torchvision transforms.
    """

    def __init__(self, transforms: Sequence[Any]):
        self.transforms = transforms;

    def __call__(self, 
        xs: Union[
                np.ndarray, 
                torch.Tensor, 
                Sequence[np.ndarray], 
                Sequence[torch.Tensor],
        ],
    ) -> Union[
            np.ndarray, 
            torch.Tensor, 
            Sequence[np.ndarray], 
            Sequence[torch.Tensor],
    ] :
    
        for t in self.transforms :
            xs = t(xs);

        return xs;


class RandomScale(object) :
    def __init__(self, 
        lim: Sequence[float] ,
    ) -> None :

        self.lim = lim;

    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :
    
        assert _is_numpy(pts);
        factor = np.random.uniform(*self.lim);
        pts[..., :3] *= factor;
        return pts;


class RandomNoise(object) :
    def __init__(self, 
        lim: float,
        rm_global_scale: bool = False,
    ) -> None :

        self.lim = _pair(lim, negate=True);
        self.rm_global_scale = rm_global_scale;

    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts);
        # select which values to perturb
        xyz = pts[..., :3];

        if self.rm_global_scale :
            ds = (xyz[:, -1, :] - xyz[:, -2, :]);
            ds = np.min(np.sqrt(np.sum(np.power(ds, 2), axis=1)));
            lim = [x * ds for x in self.lim];
        else :
            lim = self.lim;

        shape_ = xyz.shape;
        xyz = xyz.flatten();
        n_perturb = np.random.randint(1, xyz.size+1);
        mask = np.random.permutation(xyz.size)[: n_perturb];        
        xyz[mask] += np.random.uniform(*lim, n_perturb);
        pts[..., :3] = xyz.reshape(shape_);
        return pts;


class RandomTranslation(object) :
    def __init__(self, 
        xlim: Union[Sequence[float], float], 
        ylim: Union[Sequence[float], float], 
        zlim: Union[Sequence[float], float], 
    ) -> None :

        self.xlim = _pair(xlim, negate=True);
        self.ylim = _pair(ylim, negate=True);
        self.zlim = _pair(zlim, negate=True);

    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts);
        x_tr = random.uniform(*self.xlim);
        y_tr = random.uniform(*self.ylim);
        z_tr = random.uniform(*self.zlim);         

        pts[..., :3] += [x_tr, y_tr, z_tr];

        return pts;


class RandomRotation(object) :
    def __init__(self, 
        xlim: Union[Sequence[float], float] = 5.0, 
        ylim: Union[Sequence[float], float] = 5.0, 
        zlim: Union[Sequence[float], float] = 5.0,
    ) -> None :

        self.xlim = _pair(xlim, negate=True);
        self.ylim = _pair(ylim, negate=True);
        self.zlim = _pair(zlim, negate=True);

        self.xlim = _deg2rad(self.xlim);
        self.ylim = _deg2rad(self.ylim);
        self.zlim = _deg2rad(self.zlim);
        

    def __call__(self, pts: np.ndarray) -> np.ndarray :
        assert _is_numpy(pts);

        if random.random() < 0.5 :
            return pts;

        x = random.uniform(*self.xlim);
        y = random.uniform(*self.ylim);
        z = random.uniform(*self.zlim);

        sx, cx = math.sin(x), math.cos(x);
        sy, cy = math.sin(y), math.cos(y);
        sz, cz = math.sin(z), math.cos(z);

        R = np.array([ [cz*cy, sz*cy, -sy],
                       [cz*sy*sx-sz*cx, sz*sy*sx+cz*cx, cy*sx],
                       [cz*sy*cx+sz*sx, sz*sy*cx-cz*sx, cy*cx],
                    ], dtype=np.float32);

        pts[..., :3] = np.matmul(pts[..., :3], R);
        return pts;


class RandomTimeInterpolation(object) :
    def __init__(self, 
        prob: float ,
    ) -> None :

        assert 0 < prob <= 1, \
            f"Interpolation probability must be in (0, 1], got {prob}";
        self.prob = prob;


    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts);

        if random.random() > self.prob :
            return pts;

        t_shift = np.random.uniform(0, 1);
        dir_ = (np.roll(pts, -1, axis=0) - pts);
        
        pts = pts + t_shift * dir_;
        pts = pts[:-1]; # drop the last invalid frame
        return pts;


class StratifiedSample(object) :
    def __init__(self, 
        n_samples: int ,
    ) -> None :

        assert 0 < n_samples, f"#Frames to sample must be > 0, got {n_samples}";
        self.n_samples = n_samples;

    
    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts);

        n_frames = pts.shape[0];
        if self.n_samples == 1 :
            i = np.random.randint(n_frames);
            return pts[i:i+1];

        if n_frames == self.n_samples :
            return pts;
        
        if n_frames < self.n_samples :
            shape_ = pts.shape;
            # repeat last frame
            pts = np.concatenate(
                    (pts, 
                    np.repeat(pts[-1:], self.n_samples - n_frames, 0),
                    ), axis=0
            );

            return pts;

        mask = np.floor(np.linspace(0, n_frames-1, self.n_samples)).astype(np.int32);
        return pts[mask];


class CenterByIndex(object) :
    def __init__(self, 
        ind: int,
        n_hands: int = 1,
    ) -> None :

        self.ind = ind;
        self.n_hands = n_hands;

    def center_single(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        n_ind = pts.shape[1];
        assert self.ind < n_ind, \
            f"Index {self.ind} must be in [0, {n_ind-1}).";

        center_ = pts[0:1, self.ind:self.ind+1];
        pts -= center_;

        return pts;

    def __call__(self, 
        pts: np.ndarray,
    ) ->  np.ndarray :

        assert _is_numpy(pts);

        if self.n_hands == 1 :
            return self.center_single(pts);
        
        pts_left = pts[:, :21, :];
        pts_right = pts[:, 21:, :];

        is_present_left = not np.allclose(pts_left, 0);
        is_present_right = not np.allclose(pts_right, 0);
        if is_present_left and is_present_right :
            orientation = 'b';
        elif is_present_left :
            orientation = 'l';
        else :
            orientation = 'r';

        if orientation == 'b' :
            return self.center_single(pts);
        
        if orientation == 'l' :
            pts_left = self.center_single(pts_left);
        else :
            pts_right = self.center_single(pts_right);
        
        pts = np.concatenate((pts_left, pts_right), axis=1);
        return pts;


class ToTensor(object) :
    def __init__(self) :
        pass;
    
    def __call__(self, 
        pts: np.ndarray,
    ) ->  torch.Tensor :
        
        assert _is_numpy(pts);
        return torch.from_numpy(pts).to(torch.get_default_dtype());