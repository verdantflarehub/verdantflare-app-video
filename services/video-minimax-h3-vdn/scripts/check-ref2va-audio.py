"""Check upstream Ref2VA resampling with the installed Torch/TorchAudio pair."""
import torch
import torchaudio
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep

assert torch.__version__ == '2.13.0+cu129'
assert torchaudio.__version__ == '2.11.0+cu129'
for rate in (16000, 44100, 48000):
    wave = torch.sin(torch.arange(rate, dtype=torch.float32) * (2 * torch.pi * 440 / rate))[None]
    stereo = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(wave, rate, 48000, 0.5)
    assert stereo.shape == (2, 24000)
    assert torch.isfinite(stereo).all() and stereo.abs().max() > 0.5
    assert torch.equal(stereo[0], stereo[1])
print('Ref2VA audio verified: native-rate trim, mono-to-stereo, 16/44.1/48 kHz to 48 kHz')
