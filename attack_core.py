import os
import tempfile
import traceback

import torchaudio
import python_stretch as ps
import numpy as np
import torch
import cma
import random
import librosa
from pathlib import Path
from change_phase import *
from losses import MultiMelSpectrogramLoss
from denoise import *
from align_loudness import *
from app import fn


def _ensure_audio_tensor(audio):
    """
    torchaudio.save 需要 [channels, samples] 格式。
    这里做一个保险转换。
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)

    audio = audio.detach().cpu().float()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    if audio.dim() == 2:
        return audio

    if audio.dim() == 3:
        # 常见情况：[batch, channels, samples] 或 [batch, samples]
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return audio

    raise ValueError(f"Unsupported audio shape: {tuple(audio.shape)}")


def surrogate_attack(
    wav_path,
    save_dir=None,
    threshold=0.5,
    sr=22050,
    max_outer_iter=50,
    max_cma_iter=10,
):
    """
    输入:
        wav_path: 用户上传音频在服务器上的临时路径
        save_dir: 输出目录；如果不传，会自动创建临时目录

    输出:
        output_path: 生成后的 wav 文件路径，可被 Gradio 用于播放和下载
        info: 状态信息
    """

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="timbre_attack_result_")

    os.makedirs(save_dir, exist_ok=True)

    input_stem = Path(wav_path).stem
    output_path = os.path.join(save_dir, f"{input_stem}.wav")

    decoder = None
    last_audio = None
    last_best_acc = None
    last_best_recover_acc = None

    try:
        encoder, decoder = get_encoder_decoder()

        wmaudio, msg = embedding_surrogate(wav_path, encoder, decoder)

        oriaudio = wmaudio.detach().clone().cuda()

        del encoder
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        decoder = decoder.cuda().eval()

        (new_sr, wav2) = fn(wmaudio.cpu())

        target_sr = sr
        if new_sr != target_sr:
            wav2 = librosa.resample(
                y=wav2.astype(np.float32),
                orig_sr=new_sr,
                target_sr=target_sr,
                axis=-1,
            )
            new_sr = target_sr

        wmaudio = torch.from_numpy(wav2).unsqueeze(0).float().contiguous()

        bands_num_initial = 200

        current_boundaries = generate_uniform_freq_points(
            sr,
            bands_num_initial,
        )

        current_ratios = generate_random_ratios(
            bands_num_initial,
            cents_range=(60, 85),
        )

        for i in range(max_outer_iter):
            print(f"[INFO] Outer iteration: {i + 1}/{max_outer_iter}")

            new_boundaries = generate_bounded_random_cuts(
                current_boundaries,
                max_cuts=3,
            )

            new_ratios = inherit_ratios(
                current_boundaries,
                current_ratios,
                new_boundaries,
            )

            (
                audio_new,
                best_acc,
                best_recover_acc,
                optimized_ratios,
                final_boundaries,
            ) = cmaes_optimize(
                wmaudio,
                oriaudio,
                msg,
                decoder,
                max_iter=max_cma_iter,
                ratios=new_ratios,
                boundaries=new_boundaries,
                iter=i,
                threshold=threshold,
            )

            last_audio = audio_new
            last_best_acc = float(best_acc)
            last_best_recover_acc = float(best_recover_acc)

            current_boundaries = final_boundaries
            current_ratios = optimized_ratios

            print(
                f"[INFO] best_acc={last_best_acc:.4f}, "
                f"best_recover_acc={last_best_recover_acc:.4f}"
            )

            if best_acc <= threshold and best_recover_acc <= threshold:
                audio_to_save = _ensure_audio_tensor(audio_new)
                torchaudio.save(output_path, audio_to_save, sr)

                info = (
                    f"best_acc={last_best_acc:.4f}\n"
                    f"best_recover_acc={last_best_recover_acc:.4f}\n"
                    f"output={output_path}"
                )

                print("[INFO] Everything is good. Saved:", output_path)
                return output_path, info

        if last_audio is not None:
            audio_to_save = _ensure_audio_tensor(last_audio)
            torchaudio.save(output_path, audio_to_save, sr)

            info = (
                "未达到阈值，但已保存最后一轮结果。\n"
                f"best_acc={last_best_acc:.4f}\n"
                f"best_recover_acc={last_best_recover_acc:.4f}\n"
                f"output={output_path}"
            )

            print("[INFO] Saved last result:", output_path)
            return output_path, info

        raise RuntimeError("没有生成有效音频。")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        raise RuntimeError(f"处理失败: {str(e)}")

    finally:
        try:
            encoder = None
            decoder = None
            wmaudio = None
            oriaudio = None
            audio_new = None
            last_audio = None
            msg = None
            audio_to_save = None
        except Exception:
            pass

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    output_path, info = surrogate_attack(
        wav_path="realworld_google_audio_speech/speaker1-aoede-1.wav",
        save_dir="realworld_google_audio_speech_timbre_attack",
    )
    print(info)
    print("Output:", output_path)