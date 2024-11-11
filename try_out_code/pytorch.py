import os
import sys
import subprocess
import torch

need_pytorch3d = False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True

if need_pytorch3d:
    if torch.__version__.startswith("2.2.") and sys.platform.startswith("linux"):
        # Construct the version string
        pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
        version_str = "".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".", ""),
            f"_pyt{pyt_version_str}"
        ])
        
        # Install dependencies using subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fvcore", "iopath"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--no-cache-dir", 
                               "pytorch3d", "-f", f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"])
    else:
        # Install PyTorch3D from source
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/pytorch3d.git@stable'])


from utils import image_grid
import numpy as np
import torch

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)

from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
SHAPENET_PATH = ""
shapenet_dataset = ShapeNetCore(SHAPENET_PATH)