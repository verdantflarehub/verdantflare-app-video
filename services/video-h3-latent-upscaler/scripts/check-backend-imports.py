"""Image-build dependency smoke check; CPU import is not GPU capacity evidence."""
import inspect
from comfy.cli_args import args
args.cpu = True
from h3_latent_upscaler.engine import configure_comfy
configure_comfy()
import comfy.sd
import comfy.samplers
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
from h3_latent_upscaler.vendor.split_upscale import MMH3SplitUpscale
from h3_latent_upscaler.vendor.upscaler_3d import LatentResizer3D
assert 'disable_dynamic' in inspect.signature(comfy.sd.load_diffusion_model).parameters
assert callable(MiniMaxH3SigmaShift.execute)
assert callable(MMH3SplitUpscale.execute)
assert callable(LatentResizer3D.forward)
assert callable(comfy.samplers.sampler_object)
print('Pinned backend imports and entry points available (CPU only).')
