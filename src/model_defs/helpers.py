import torch 
import torch.nn as nn
import numpy as np
import numbers 

from typing import Optional, Sequence, Union, Callable


def _pair(x) :
    if isinstance(x, numbers.Number) :
        return (x, x)  

    assert len(x)==2
    return x  

def _isinstance_pair(x, type_) :
    return isinstance(x[0], type_) and isinstance(x[1], type_)


def print_n_params(model) :
    ndigits = lambda x : len(str(x).split('.')[0])
    params = filter(lambda p: p.requires_grad, model.parameters())
    nparams = sum([np.prod(p.size()) for p in params])
    if ndigits(nparams) < 6 :
        div_ = 1e3
        quant_ = 'K'
    else :
        div_ = 1e6
        quant_ = 'M'
    nparams = str(round(nparams/div_, 2)) + quant_
    print(f"Number of trainable params = {nparams}")


def get_norm_layer_from_type(
    type_: str,
    dim: int,
    channels: int,
) -> Optional[torch.nn.Module] :

    assert 1<=dim<=2, f"dim must be in [1,2], got {dim}"

    if type_ == None :
        return None
    elif type_ == 'batch' :
        norm_layer = nn.BatchNorm1d if dim==1 else nn.BatchNorm2d
        return norm_layer(channels)
    elif type_ == 'instance' :
        norm_layer = nn.InstanceNorm1d if dim==1 else nn.InstanceNorm2d
        return norm_layer(channels)
    else :
        raise NotImplementedError


def get_act_layer_from_type(
    type_: str, 
    inplace: bool = True,
) -> Optional[torch.nn.Module] :

    if type_ == None :
        return None
    elif type_ == 'relu' :
        return nn.ReLU(inplace)
    elif type_ == 'hard_swish' :
        return nn.Hardswish
    elif type_ == 'hard_sigmoid' :
        return nn.Hardsigmoid        
    else :
        raise NotImplementedError


def conv2d(
    in_channels: Union[int, Sequence], 
    out_channels: Union[int, Sequence],
    kernel_size: Union[int, Sequence],
    stride: Union[int, Sequence] = 1,
    padding: Optional[Union[int, Sequence]] = None,
    groups: int = 1,
    dilation: Union[int, Sequence] = 1,
    bias: Optional[bool] = None,
) -> nn.Conv2d :

    if padding is None :
        padding = (kernel_size - 1) // 2 * dilation
    
    bias = True if bias else False

    return torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
    )


def conv2d_norm_act(
    in_channels: int, 
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: Optional[int] = None,
    groups: int = 1,
    dilation: int = 1,
    norm_type: Optional[str] = None,
    act_type: Optional[str] = None,
    inplace: Optional[bool] = True,
    bias: Optional[bool] = None,
) :
    """Source: https://github.com/pytorch/vision/blob/44252c81c877075c8415ea242a501fa227d0d8af/torchvision/ops/misc.py#L69"""

    norm_layer = get_norm_layer_from_type(norm_type)
    act_layer = get_act_layer_from_type(act_type) 

    if padding is None :
        padding = (kernel_size - 1) // 2 * dilation
    if bias is None :
        bias = (not isinstance(norm_layer, torch.nn.BatchNorm2d))

    layers = [
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    ]

    if norm_layer is not None :
        layers.append(norm_layer(out_channels))
    
    if act_layer is not None :
        act_params = {} if inplace is None else {'inplace': inplace}
        layers.append(act_layer(**act_params))

    return layers


@torch.no_grad()
def init_weights(
        m: nn.Module, 
        init_method: str='xavier_uniform', 
) -> None :

    init_method_all = ('xavier_uniform', )

    init_method = init_method.lower()
    assert init_method in init_method_all, \
        f"init_method {init_method} must be one of {init_method_all}"

    if init_method == 'xavier_uniform' :
        if isinstance(m, nn.Conv2d) :
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
                
        elif isinstance(m, nn.BatchNorm2d) :
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)  

    else :
        raise NotImplementedError