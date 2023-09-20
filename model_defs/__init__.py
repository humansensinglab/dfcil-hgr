try :
    from . import dg_sta
    from . import dg_sta_contrastive
    from .helpers import print_n_params
except :
    import dg_sta
    import dg_sta_contrastive
    from helpers import print_n_params

def get_model(cfg) :
    assert cfg.name in globals(), \
        f"Model {cfg.name} not found.";

    if cfg.name == 'dg_sta' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            dropout=cfg.dropout, 
        );

    if cfg.name == 'dg_sta_contrastive' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            d_head_contrastive=cfg.d_head_contrastive,
            dropout=cfg.dropout, 
        );        

    else :
        raise NotImplementedError;


if __name__ == "__main__" :
    from helpers import print_n_params
    from easydict import EasyDict as edict

    import yaml
    import torch

    import sys; sys.path.append('..');
    from configs.datasets import hgr_shrec_2017

    root_dir = '/data/datasets/agr/shrec2017';
    cfg_file = '../configs/params/initial.yaml';
    split_type = 'specific';
    cfg_data = hgr_shrec_2017.Config_Data(root_dir);

    with open(cfg_file, 'rb') as f :
        cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader));

    model = get_model(edict({
            'n_classes': cfg_data.get_n_classes(split_type),
            **cfg_params.model,
    }));
    model = model.cuda(0);
    print(model);
    print_n_params(model);
    
    bs, seq_len, n_joints, in_channels = 4, 8, 22, 3;
    x = torch.rand(bs, seq_len, n_joints, in_channels).cuda(0);
    out = model(x);
    print(x.shape, out.shape);