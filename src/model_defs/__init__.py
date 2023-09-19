try :
    from . import dg_sta
    from . import dg_sta_BN
    from .helpers import print_n_params
except :
    import dg_sta
    import dg_sta_BN
    from helpers import print_n_params

def get_model(cfg) :
    assert cfg.name in globals(), \
        f"Model {cfg.name} not found."

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
        )

    if cfg.name == 'dg_sta_BN' :
        print('Using dg_sta_BN model.')
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            dropout=cfg.dropout, 
        )

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
        )        

    else :
        raise NotImplementedError


