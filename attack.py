import torchaudio
import python_stretch as ps
import numpy as np
from change_phase import *
from losses import MultiMelSpectrogramLoss
from denoise import *
from align_loudness import *
import torch
import cma
import random
import os
import librosa
from app import fn



def surrogate_attack(
    wav_path='realworld_google_audio_speech/speaker1-aoede-1.wav',
    save_dir='realworld_google_audio_speech_timbre_attack',
):

    encoder, decoder = get_encoder_decoder()
    threshold = 0.5
    sr = 22050
    wmaudio, msg = embedding_surrogate(wav_path, encoder, decoder)

    oriaudio = wmaudio.detach().clone().cuda()

    del encoder
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    decoder = decoder.cuda().eval()

    _, (new_sr, wav2) = fn(wmaudio.cpu())


    target_sr= 22050
    if new_sr != target_sr:
        wav2 = librosa.resample(
            y=wav2.astype(np.float32),
            orig_sr=new_sr,
            target_sr=target_sr,
            axis=-1
        )
        new_sr = target_sr
    wmaudio = torch.from_numpy(wav2).unsqueeze(0).float().contiguous()

    flag = 0


    bands_num_initial = 200
    current_boundaries = generate_uniform_freq_points(sr, bands_num_initial) # 这个地方改成random的
    current_ratios = generate_random_ratios(bands_num_initial,  cents_range=(60, 85))

    for i in range(50):

        new_boundaries = generate_bounded_random_cuts(current_boundaries, max_cuts = 3)
        new_ratios = inherit_ratios(current_boundaries, current_ratios, new_boundaries)

        audio_new, best_acc, best_recover_acc, optimized_ratios, final_boundaries = cmaes_optimize(wmaudio, oriaudio,msg, decoder, max_iter = 10, ratios=  new_ratios, boundaries = new_boundaries, iter = i, threshold=0.5)

        current_boundaries = final_boundaries
        current_ratios = optimized_ratios
        if best_acc<=threshold and best_recover_acc<=threshold:
            flag = 1
            print('every thing is good break')
            torchaudio.save(f'{save_dir}/wav.wav', audio_new.cpu(), 22050)
            break
    
        if best_acc<=threshold and best_recover_acc<=threshold:
            if flag ==0:
                torchaudio.save(f'{save_dir}/wav.wav', torch.tensor(audio_new), 22050)
                print('every good')
            break

if __name__ == '__main__':
    surrogate_attack()