from typing import Optional

from easydict import EasyDict as edict
import math

try :
    from . import cross_entropy
    from . import contrastive
    from . import distillation
    from . import bce
    from . import abd_distillation
    from . import rkd
    from . import hkd
    from . import rdfcil_local_cross_entropy
    from . import abd_local_cross_entropy
except :
    import cross_entropy    
    import contrastive
    import distillation
    import bce
    import abd_distillation
    import rkd
    import hkd
    import rdfcil_local_cross_entropy
    import abd_local_cross_entropy


def _dict_to_list(d) :
    max_k = max(d.keys())
    assert len(d) == max_k + 1
    return [d[k] for k in range(max_k+1)]


def get_losses(cfg: dict, n_classes: int, class_freq: Optional[dict] = None) :
    criteria = edict()
    for lname in cfg :
        assert lname in globals(), f"{lname} not found."
        criteria[lname] = edict()
        criteria[lname].weight = cfg[lname].weight
        if lname == 'cross_entropy' :
            weights = None
            if cfg[lname].class_weight == 'class_freq' :
                raise NotImplementedError
                
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                n_classes,
                cfg[lname].ignore,
                weights,
            )
        
        elif lname == 'contrastive' :
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                cfg[lname].n_views,
                cfg[lname].is_supervised,
                cfg[lname].temperature,
            )  

        elif lname == 'distillation' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()  

        elif lname == 'abd_distillation' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()  

        elif lname == 'bce' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                cfg[lname].reduction
            )    
            
        elif lname == 'rkd' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                cfg[lname].in_dim1,
                cfg[lname].in_dim2,
                cfg[lname].proj_dim,
            )
        elif lname == 'hkd' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()    

        elif lname == 'rdfcil_local_cross_entropy' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()   

        elif lname == 'abd_local_cross_entropy' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()    
       
        else : 
            raise NotImplementedError

    return criteria




