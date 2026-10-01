# Thesis: "Concatenation Fusion non pre-trained"
# concatenation + conv + BN + ReLU (MapTR's ConvFuser), trained from scratch
_base_ = ['./maptr_tiny_r50_24e_aerial_cwl.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            aerial_fuser=dict(
                _delete_=True,
                type='ConvFuser',
                in_channels=[256, 256],
                out_channels=256))))

# train all parameters, no MapTR initialization
load_from_maptr = None
trainable_modules = None
