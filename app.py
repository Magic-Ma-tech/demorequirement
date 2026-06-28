import gradio as gr
import torch
import torchaudio
import gc
from resemble_enhance.enhancer.inference import denoise, enhance, load_enhancer
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def fn(dwav, sr=22050, solver="Midpoint", nfe=64, tau=0.5, denoising=True):
    # if path is None:
    #     return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    # dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    # wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    try:
        del wav1
    except Exception:
        pass

    try:
        load_enhancer.cache_clear()
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return (new_sr, wav2)

    

