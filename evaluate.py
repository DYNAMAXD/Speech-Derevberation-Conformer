import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from tqdm import tqdm 
from training import FastConformer, EvalConfig

 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# FEATURE EXTRACTION 
class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.SAMPLE_RATE,
            n_fft=cfg.N_FFT,
            win_length=cfg.WIN_LENGTH,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS,
            f_min=cfg.F_MIN,
            f_max=cfg.F_MAX,
            power=2.0,
        )

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.cfg.SAMPLE_RATE:
            wav = T.Resample(sr, self.cfg.SAMPLE_RATE)(wav)
        return wav

    def pre_emphasis(self, x):
        y = torch.zeros_like(x)
        y[:, 0] = x[:, 0]
        y[:, 1:] = x[:, 1:] - self.cfg.PREEMPHASIS * x[:, :-1]
        return y

    def audio_to_logmel(self, wav, rt60):
        wav = self.pre_emphasis(wav)
        mel = self.mel(wav)
        logmel = torch.log(mel + 1e-10)
        logmel = (logmel - self.cfg.DATA_MEAN) / self.cfg.DATA_STD

        T_len = logmel.shape[-1]
        rt = torch.full((1, 1, T_len), rt60)
        return torch.cat([logmel, rt], dim=1)

    def denorm(self, x):
        return x * self.cfg.DATA_STD + self.cfg.DATA_MEAN

 
# LOAD HIFIGAN  
def load_hifigan(device):
    print("\nLoading HiFi-GAN vocoder (TorchHub)")
    print("• First run will download weights")
    print("• CPU mode is slow but supported\n")

    with tqdm(
        total=1,
        desc="Downloading / Loading HiFi-GAN",
        bar_format="{l_bar}{bar}| {elapsed}",
    ) as pbar:
        hifigan, _, denoiser = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_hifigan",
            map_location=device,
        )
        pbar.update(1)

    hifigan = hifigan.to(device).eval()
    denoiser = denoiser.to(device).eval()

    print("✓ HiFi-GAN ready\n")
    return hifigan, denoiser

 
# MAIN 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--rt60", type=float, default=2.0)
    parser.add_argument("--out", default="evaluation_results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Speech Dereverberation Evaluation")
    print("=" * 60)
    print(f"Device: {DEVICE}\n")

    if DEVICE.type == "cpu":
        print("Running on CPU")
        print("HiFi-GAN vocoding will be slow\n")

    # ---------------- LOAD CHECKPOINT ----------------
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)

    cfg = EvalConfig()
    for k, v in ckpt["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    model = FastConformer(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    extractor = FeatureExtractor(cfg)

    # ---------------- LOAD HIFIGAN ----------------
    hifigan, denoiser = load_hifigan(DEVICE)

    # ---------------- PROCESS AUDIO ----------------
    print("Loading audio...")
    wav = extractor.load_audio(args.audio)
    feats = extractor.audio_to_logmel(wav, args.rt60).to(DEVICE)
    seq_lens = torch.tensor([feats.shape[-1]], device=DEVICE)

    print("Running model inference...")
    with torch.no_grad():
        out_feats = model(feats, seq_lens)

    # ---------------- METRICS ----------------
    inp = feats[:, :cfg.N_MELS]
    outp = out_feats

    mse = F.mse_loss(outp, inp).item()
    mae = F.l1_loss(outp, inp).item()
    energy_change = (
        torch.mean(outp ** 2) - torch.mean(inp ** 2)
    ) / torch.mean(inp ** 2) * 100

    print("\nEvaluation Metrics")
    print("-" * 30)
    print(f"MSE           : {mse:.6f}")
    print(f"MAE           : {mae:.6f}")
    print(f"Energy Change : {energy_change:+.2f}%")

    # ---------------- HIFIGAN VOCODING ----------------
    print("\nReconstructing audio with HiFi-GAN...")
    mel_denorm = extractor.denorm(outp).float()
    audio = hifigan(mel_denorm).squeeze(1)
    audio = denoiser(audio, 0.005)

    audio = audio.squeeze(1)

    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = torch.clamp(audio, -1.0, 1.0)

    audio_np = audio[0].cpu().numpy().astype("float32")

    sf.write(
        out_dir / "output.wav",
        audio_np,
        cfg.SAMPLE_RATE,
        subtype="PCM_16",
    )
 
    # ---------------- SAVE SUMMARY ----------------
    summary = {
        "mse": mse,
        "mae": mae,
        "energy_change_percent": float(energy_change),
        "rt60": args.rt60,
        "device": DEVICE.type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation complete")
    print(f"Results saved to: {out_dir.resolve()}") 


if __name__ == "__main__":
    main()