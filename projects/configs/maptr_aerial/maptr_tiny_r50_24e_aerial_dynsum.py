# Thesis: "Dynamic Summed Fusion pre-trained FS: BCWH"
_base_ = ['./maptr_tiny_r50_24e_aerial_cwl.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            aerial_fuser=dict(
                _delete_=True,
                type='DynamicWeightSumFuser',
                in_channels=256,
                out_channels=256))))
