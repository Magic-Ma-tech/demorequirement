import os
import soundfile as sf
from losses import MultiMelSpectrogramLoss
import os
import torch
import yaml
import logging
import argparse
import warnings
import numpy as np
from rich.progress import track
from torch.utils.data import DataLoader
import torchaudio
from timbre_watermarking.model.conv2_mel_modules import Encoder, Decoder, Discriminator
# import tensorflow as tf
import random
from wavmark_main.src import wavmark
import soundfile as sf
import python_stretch as ps
import pyworld
from scipy.ndimage import gaussian_filter1d
import cma
from denoise import *
from align_loudness import *
from pathlib import Path




warnings.filterwarnings("ignore")

# set seeds
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
batch_size = 1
msg_length = 10


mel_loss = MultiMelSpectrogramLoss()
aligner = AudioVolumeAligner()

# BASE_DIR = Path(__file__).resolve().parent

# PROCESS_CONFIG = BASE_DIR / "timbre_watermarking" / "config_wm" / "process.yaml"
# MODEL_CONFIG = BASE_DIR / "timbre_watermarking" / "config_wm" / "model.yaml"


BASE_DIR = Path(__file__).resolve().parent

# 兼容两种结构：
# 1. /root/change_phase.py + /root/timbre_watermarking/...
# 2. /root/timbre_watermarking/change_phase.py + /root/timbre_watermarking/config_wm/...
if (BASE_DIR / "config_wm").exists():
    WM_ROOT = BASE_DIR
else:
    WM_ROOT = BASE_DIR / "timbre_watermarking"

PROCESS_CONFIG = WM_ROOT / "config_wm" / "process.yaml"
MODEL_CONFIG = WM_ROOT / "config_wm" / "model.yaml"
TRAIN_CONFIG = WM_ROOT / "config_wm" / "train.yaml"
EXPERIMENTS_DIR = WM_ROOT / "experiments_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_project_path(path_value):
    """
    把本地绝对路径，例如：
    /home/binhaoma/local/generation_model_Bench/timbre_watermarking/xxx
    映射成 Modal 里的：
    /root/timbre_watermarking/xxx
    """
    p = Path(path_value).expanduser()

    if p.exists():
        return p

    parts = list(p.parts)

    if "timbre_watermarking" in parts:
        idx = parts.index("timbre_watermarking")
        relative_tail = Path(*parts[idx + 1:])
        mapped = WM_ROOT / relative_tail
        return mapped

    if p.is_absolute():
        return p

    candidate_1 = WM_ROOT / p
    if candidate_1.exists():
        return candidate_1

    candidate_2 = BASE_DIR / p
    if candidate_2.exists():
        return candidate_2

    return candidate_1


def load_yaml(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"BASE_DIR={BASE_DIR}\n"
            f"WM_ROOT={WM_ROOT}\n"
            f"Files under WM_ROOT: {os.listdir(WM_ROOT) if WM_ROOT.exists() else 'WM_ROOT does not exist'}"
        )

    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_encoder_decoder():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("WM_ROOT:", WM_ROOT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--process_config",
        type=str,
        default=str(PROCESS_CONFIG),
        help="Path to process.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=str(MODEL_CONFIG),
        help="Path to model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        default=str(TRAIN_CONFIG),
        help="Path to train.yaml",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=str(EXPERIMENTS_DIR),
        help="Path to save results",
    )

    args = parser.parse_args([])

    process_config_path = resolve_project_path(args.process_config)
    model_config_path = resolve_project_path(args.model_config)
    train_config_path = resolve_project_path(args.train_config)

    print("process_config:", process_config_path)
    print("model_config:", model_config_path)
    print("train_config:", train_config_path)

    process_config = load_yaml(process_config_path)
    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)

    win_dim = process_config["audio"]["win_len"]

    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]

    msg_length = train_config["watermark"]["length"]

    encoder = Encoder(
        process_config,
        model_config,
        msg_length,
        win_dim,
        embedding_dim,
        nlayers_encoder=nlayers_encoder,
        attention_heads=attention_heads_encoder,
    ).to(device)

    decoder = Decoder(
        process_config,
        model_config,
        msg_length,
        win_dim,
        embedding_dim,
        nlayers_decoder=nlayers_decoder,
        attention_heads=attention_heads_decoder,
    ).to(device)

    path_model = resolve_project_path(model_config["test"]["model_path"])
    model_name = model_config["test"].get("model_name", None)
    index = model_config["test"]["index"]

    if not path_model.exists():
        raise FileNotFoundError(
            f"Model checkpoint directory not found: {path_model}\n"
            f"Original model_path in yaml: {model_config['test']['model_path']}\n"
            f"WM_ROOT={WM_ROOT}"
        )

    model_list = os.listdir(path_model)

    # 可选：如果里面有非 checkpoint 文件，可以在这里过滤
    model_list = [
        x for x in model_list
        if not x.startswith(".")
    ]

    if len(model_list) == 0:
        raise FileNotFoundError(f"No checkpoint files found in: {path_model}")

    model_list = sorted(
        model_list,
        key=lambda x: os.path.getmtime(os.path.join(path_model, x)),
    )

    model_path = os.path.join(path_model, model_list[index])

    print("loading checkpoint:", model_path)
    model = torch.load(model_path, map_location=device)

    logging.info("model <<{}>> loaded".format(model_path))

    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)

    encoder.eval()
    decoder.eval()

    return encoder, decoder



def global_recover(attack_audio, current_msg, wmaudio, decoder, sr=22050, step_up_floor = 1.10, step_down_floor = 0.90):
    # _,decoder = get_encoder_decoder()
    mel_loss = MultiMelSpectrogramLoss()
    # scalar_ratio = 1
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 22050)
    min_mel = 100
    final_acc = 1
    recover_ratio =2
    scalar_ratio_list = np.linspace(step_down_floor, step_up_floor, 100)
    min_audio = attack_audio

    for _, ratio in enumerate(scalar_ratio_list):

        stretch.setFreqMap(custom_map_factory(ratio, sr))
        global_recover_audio = stretch.process(attack_audio)

        min_len = min(global_recover_audio.shape[1], wmaudio.shape[1])
        global_recover_audio = torch.tensor(global_recover_audio)
        # print('type is -00-----------', type(global_recover_audio))
        a = global_recover_audio[:, :min_len]  # shape [1, min_len]
        b = wmaudio[:, :min_len]           # shape [1, min_len]

    
        mel = mel_loss(a, b)


        recover_acc = query(current_msg, global_recover_audio, decoder, watermark='timbre')

        if mel<min_mel:
            min_mel = mel
            min_audio = global_recover_audio
            final_acc = recover_acc
            recover_ratio = ratio

    return final_acc, min_audio, recover_ratio



def query(msg, wav, decoder, watermark = 'timbre'):
    # 频率映射
    # stretch.setFreqMap(custom_map_factory(ratio, sr, n_bins))
    # audio_processed = stretch.process(wav)
    # _, decoder = get_encoder_decoder()
    with torch.no_grad():
        if watermark == 'timbre':
            if type(wav) == np.ndarray:
                wav = torch.from_numpy(wav)

            # 解码
            with torch.no_grad():
                # decoded = decoder.test_forward(wav.unsqueeze(0).cuda())

                device = next(decoder.parameters()).device
                decoded = decoder.test_forward(wav.unsqueeze(0).to(device))

            decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
            decoder_acc = round(decoder_acc.item(),2)
        elif watermark == 'audioseal':
            result, msg = decoder.detect_watermark(torch.tensor(wav).unsqueeze(0), sample_rate=16000, message_threshold=0.5)
            decoder_acc = result
        
        elif watermark == 'robustdnn':
            # if not hasattr(query, "_tf_inited"):
            #     gpus = tf.config.list_physical_devices('GPU')
            #     for gpu in gpus:
            #         tf.config.experimental.set_memory_growth(gpu, True)
            #     query._tf_inited = True
            from dnn_audio_watermarking.main import detect

            if wav.shape[0] == 1:
                wav = wav.squeeze(0)


            TARGET_LENGTH = 33226
            signal = tf.convert_to_tensor(wav, dtype=tf.float32)
            # 检查信号长度并进行调整
            if len(signal) < TARGET_LENGTH:
                # 填充：如果信号太短，就在末尾用0填充
                padding_length = TARGET_LENGTH - len(signal)
                signal = np.pad(signal, (0, padding_length), 'constant')

            elif len(signal) > TARGET_LENGTH:
                # 裁剪：如果信号太长，就截取前 TARGET_LENGTH 个样本
                signal = signal[:TARGET_LENGTH]

            # signal = tf.convert_to_tensor(wav, dtype=tf.float32)    
            signal = tf.expand_dims(signal, axis=0)  # shape [1, num_samples]

            watermark = detect(decoder, signal)
            

            bits = np.where(watermark >= 0.5, 1, 0)
            matches = np.sum(bits == msg)
            total = len(bits[0])
            decoder_acc = np.round(matches / total,2)
        elif watermark == 'audiomarknet':

            if wav.shape[0] == 1:
                wav = wav.squeeze(0)

            int_number = len(wav)/16000
            need_detect = int(int_number)
            with torch.no_grad():
                x = wav
                wav_secs = decoder.split_waveform(x)

                logits = decoder.decoder(decoder.spec_for_classificiation(wav_secs)[..., decoder.min_freq_idx: decoder.max_freq_idx, :])
            pred_wm = (logits > 0.0).long()
            pred_wm = pred_wm[:need_detect,:]
            list_acc = []
            for one_wm in pred_wm:
                decoder_acc = (one_wm == msg).sum() / one_wm.numel()
                list_acc.append(decoder_acc.item())
                
            # print('acc is --------------', decoder_acc)
            decoder_acc = max(list_acc)
            # print(f'bestacc is {decoder_acc}')
            # print('decoder_acc is ', decoder_acc)
        elif watermark == 'wavmark':
            payload_decoded, _ = wavmark.decode_watermark(decoder, wav, show_progress=True)
            if payload_decoded is None:
                decoder_acc = 0
            else:
                decoder_acc=((payload_decoded > 0) == (msg > 0)).float().mean()
        
        elif watermark == 'silentcipher':
            result = decoder.decode_wav(np.squeeze(wav), 44100, phase_shift_decoding=False)
            if result['status'] == False:
                decoder_acc = 0
            else:
                decoder_acc =np.mean(np.array(result['messages'][0]) == np.array([123, 234, 111, 222, 11]))
        elif watermark == 'collaborative':
            wav = torch.tensor(wav).unsqueeze(0)
            result1, result = decoder(wav, wav)
            decoder_acc = np.round((1-result).item(),2)
            # print('1-result is', decoder_acc)
            # clean -------------------------------------

        


    return decoder_acc




def custom_map_factory(ratio_or_array, sr, n_bins=1000, freq_points=None):
    """
    ratio_or_array: scalar 或 array
    sr: 采样率
    n_bins: 分段数量
    freq_points: (optional) 自定义分段边界
    """
    if np.isscalar(ratio_or_array):
        def freq_map(f):
            return f * ratio_or_array
        return freq_map
    
    ratios = np.asarray(ratio_or_array)
    assert len(ratios) == n_bins, f"Length of ratios should be {n_bins}"

    # 如果没提供，就用均匀分段
    if freq_points is None:
        freq_points = np.linspace(0, sr/2, n_bins+1)
    else:
        freq_points = np.asarray(freq_points)
        assert len(freq_points) == n_bins+1, f"Length of freq_points should be {n_bins+1}"

    def freq_map(f):
        freq_hz = f * sr
        idx = np.searchsorted(freq_points, freq_hz, side='right') - 1
        idx = np.clip(idx, 0, n_bins-1)
        return f * ratios[idx]
    
    return freq_map





def global_recover_acc(attack_audio, current_msg,decoder, wm_type, sr=22050, step_up_floor = 1.05, step_down_floor = 0.95):


    # scalar_ratio = 1
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, sr)
    # final_acc = 0
    scalar_ratio_list = np.linspace(step_down_floor, step_up_floor, 100)
    minratio = 1
    ori_acc = query(current_msg, attack_audio, decoder, watermark=wm_type)
    print('original Acc is ', ori_acc)
    final_acc = ori_acc
    min_audio = attack_audio
    for _, ratio in enumerate(scalar_ratio_list):

        stretch.setFreqMap(custom_map_factory(ratio, sr))
        global_recover_audio = stretch.process(attack_audio)

        recover_acc = query(current_msg, global_recover_audio, decoder, watermark=wm_type)
        if recover_acc> final_acc:
            final_acc = recover_acc
            min_audio = global_recover_audio

            minratio = ratio
    return final_acc, min_audio, minratio


def generate_random_ratios(n_bins, cents_range=(-85, 85)): # yige array

    ratios = []
    for _ in range(n_bins):
        sign = 1
        cents = np.random.uniform(cents_range[0], cents_range[1])
        final_cents = sign * cents
        ratio = cents_to_ratio(final_cents)
        ratios.append(ratio)
    return np.array(ratios)


def cents_to_ratio(cents):
    """把cents转为频率倍数比例"""
    return 2 ** (cents / 1200)



def generate_uniform_freq_points(sr, n_bins):
    # 生成从0到 sr/2 的均匀分割点
    freq_points = np.linspace(0, sr / 2, n_bins + 1)
    return freq_points

def generate_bounded_random_cuts(boundaries, max_cuts=3):
    """
    顺序遍历每个 band，根据宽度概率决定是否切割，直到切满 max_cuts 次。
    """
    refined_boundaries = np.array(boundaries)
    widths = np.diff(refined_boundaries)
    
    # 1. 计算每个 band 的概率 (宽度 / 最大宽度)
    max_w = np.max(widths)
    probs = widths / max_w
    
    cuts_made = 0
    new_points = []
    
    # 2. 顺序遍历每一个现有的 Band
    for i in range(len(widths)):
        # 如果已经切满了，直接收工
        if cuts_made >= max_cuts:
            break
            
        # 根据当前 Band 的概率进行随机判定
        # np.random.rand() 生成 0-1 之间的数，如果小于 prob 就切
        if np.random.rand() < probs[i]:
            low = refined_boundaries[i]
            high = refined_boundaries[i+1]
            
            # 在这个 Band 内部随机选一个点
            new_pt = np.random.uniform(low, high)
            new_points.append(new_pt)
            
            cuts_made += 1
    # 3. 将所有新点加入并重新排序
    final_boundaries = np.sort(np.append(refined_boundaries, new_points))
    
    return final_boundaries

def inherit_ratios(old_boundaries, old_ratios, new_boundaries):
    """
    根据新旧边界的对应关系，让新的 ratios 继承旧的值。
    """
    old_boundaries = np.array(old_boundaries)
    new_ratios = []
    
    # 遍历每一个新生成的频带区间
    for i in range(len(new_boundaries) - 1):
        # 取新区间的中点
        mid_point = (new_boundaries[i] + new_boundaries[i+1]) / 2.0
        
        # 找到这个中点在旧边界中属于第几个 index
        # np.searchsorted 会返回 mid_point 应该插入的位置
        idx = np.searchsorted(old_boundaries, mid_point) - 1
        
        # 处理边界情况：确保索引在合法范围内
        idx = max(0, min(idx, len(old_ratios) - 1))
        
        # 继承旧的 ratio
        new_ratios.append(old_ratios[idx])
    
    return new_ratios


def objective_function(ratios, ori_audio, oriaudio_2, current_msg, decoder, sr, n_bins, boundaries, iter = 10):
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, sr)
    stretch.setFreqMap(custom_map_factory(ratios, sr, n_bins, boundaries))
    
    audio = stretch.process(ori_audio)
    # audio = torch.tensor(denoise_mmse_stsa(audio)).unsqueeze(0)
    audio = torch.tensor(audio)
    
    ori_audio_d = oriaudio_2

    print('shape is, ', audio.shape, 'ori shape is ', ori_audio_d.shape)

    # audio = aligner.align_by_peak(audio, ori_audio)

    acc = query(current_msg, audio, decoder)
    
    audio = audio.cpu()
    ori_audio_d = ori_audio_d.cpu()
    mel = mel_loss(audio, ori_audio_d)


    # if you want recover open this 
    recover_acc, _, _ = global_recover(
    current_msg=current_msg, attack_audio=audio, wmaudio=ori_audio_d, decoder=decoder
)
    loss = acc + mel + recover_acc 
    
    print(f"Current loss: {loss}, acc: {acc}, Mel_loss: {mel}, recover_acc: {recover_acc}")
    
    return (loss, {'acc': acc, 'mel_loss': mel, 'min_audio': audio, 'recover_acc':recover_acc, 'boundaries':boundaries})


threshold = 0.5
def cmaes_optimize(ori_audio, oriaudio_2,current_msg, decoder, ratios, boundaries, sr=22050, max_iter=5000, step_up_floor=1.05, step_down_floor=1, iter=10, threshold=0.5): #0.95
    # 1.05- 0.98
    # boundaries = generate_uniform_freq_points(sr, n_bins)
    initial_ratios = ratios
    initial_ratios = np.clip(initial_ratios, step_down_floor, step_up_floor)
    initial_sigma = 0.1

    es = cma.CMAEvolutionStrategy(
        initial_ratios, initial_sigma, 
        {'bounds': [step_down_floor, step_up_floor], 'maxfevals': max_iter}
    )

    # 建立全局最优追踪
    best_overall_loss = float('inf')
    best_overall_audio = None
    best_overall_ratios = None
    
    while not es.stop():
        solutions = es.ask()
        losses = []
        extra_data = []
        
        for raw_ratios in solutions:
            
            # raw_ratios = push_away_from_one(raw_ratios)
            loss, data = objective_function(raw_ratios, ori_audio, oriaudio_2, current_msg, decoder, sr, len(boundaries)-1, boundaries, iter = iter)
            
            # 确保 loss 是 float
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            
            losses.append(loss_val)
            extra_data.append(data)
            
            # --- 关键：手动记录历史全局最优 ---
            if loss_val < best_overall_loss:
                best_overall_loss = loss_val
                best_overall_audio = data['min_audio']
                best_recover_acc = data['recover_acc']

                best_acc = data['acc']
                boundaries = data['boundaries']
                best_overall_ratios = raw_ratios # 记录此时的比例

            if data['acc'] <= threshold and data['recover_acc']<=threshold:
                best_overall_loss = loss_val
                best_overall_audio = data['min_audio']
                best_recover_acc = data['recover_acc']

                best_acc = data['acc']
                boundaries = data['boundaries']
                best_overall_ratios = raw_ratios # 记录此时的比例
                return best_overall_audio, best_acc, best_recover_acc, best_overall_ratios, boundaries
            
        
        if best_acc<=threshold and best_recover_acc<=threshold:
            return best_overall_audio, best_acc, best_recover_acc, best_overall_ratios, boundaries
        
        es.tell(solutions, losses)
        print(f"Iter: {es.countiter}, Global Best Loss: {best_overall_loss}, Sigma: {es.sigma:.4f}")

    # 返回真正的全局最优
    return best_overall_audio, best_acc, best_recover_acc, best_overall_ratios, boundaries


def embedding_surrogate(wav_path, encoder, decoder):
    batch_size = 1
    msg_length = 10

    encoder = encoder.cuda().eval()
    decoder = decoder.cuda().eval()

    msg = np.ones([batch_size, 1, msg_length], dtype=np.float32)
    msg = torch.from_numpy(msg).float() * 2 - 1
    msg = msg.cuda().contiguous()

    wav, sr = librosa.load(wav_path, sr=22050)
    wav = wav[:sr * 10]

    wav_matrix = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).cuda().contiguous()

    with torch.no_grad():
        encoded, carrier_watermarked = encoder.test_forward(wav_matrix, msg)

        decoded = decoder.test_forward(encoded)

    print('encoded shape is ---', encoded.shape)
    print('decoded shape is ---', decoded.shape)
    print('msg shape is ---', msg.shape)

    decoder_acc = (decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
    print('decoder_acc is ', decoder_acc)

    wm_audio = encoded.detach().cpu().squeeze(0).contiguous()
    msg = msg.detach().cuda().contiguous()

    print("return wm_audio:", wm_audio.shape, wm_audio.dtype, wm_audio.device)
    print("return msg:", msg.shape, msg.dtype, msg.device)

    return wm_audio.float(), msg
