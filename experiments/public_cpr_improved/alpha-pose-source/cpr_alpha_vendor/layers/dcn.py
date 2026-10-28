from torchvision.ops import DeformConv2d
class DeformConv(DeformConv2d):
    def __init__(self, *args, deformable_groups=1, **kwargs):
        if deformable_groups != 1: raise ValueError("Unsupported DCN grouping")
        super().__init__(*args, **kwargs)
class ModulatedDeformConv(DeformConv):
    def forward(self, *args, **kwargs):
        raise ValueError("Modulated DCN is not used by the verified model")
