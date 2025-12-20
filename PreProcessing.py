import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import time
import json
from datetime import datetime
import sys
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
class Config:
    DRIVE_PATH = "/content/drive/MyDrive"
    DATASET_PATH = os.path.join(DRIVE_PATH, "dataset")
    RAW_PATH = os.path.join(DATASET_PATH, "finalRawSpeech")
    CLEAN_PATH = os.path.join(DATASET_PATH, "openSLR", "fullCleanSpeech")
    MAPPING_PATH = os.path.join(DATASET_PATH, "mapping", "convolution_mapping_full.csv")
    PREPROCESSED_PATH = os.path.join(DATASET_PATH, "preProcessedSpeech")
    
    SAMPLE_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    N_MELS = 80
    F_MIN = 0.0
    F_MAX = 8000
    PREEMPHASIS = 0.97
    
    BATCH_SIZE = 16
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    VISUALIZATION_SAMPLES = 2
    VISUALIZATION_DIR = os.path.join(DRIVE_PATH, "visualizations")
    
    def __init__(self):
        os.makedirs(self.PREPROCESSED_PATH, exist_ok=True)
        os.makedirs(os.path.join(self.PREPROCESSED_PATH, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.PREPROCESSED_PATH, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.PREPROCESSED_PATH, "test"), exist_ok=True)
        os.makedirs(self.VISUALIZATION_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Progress Tracker for Resume Capability
# ----------------------------------------------------------------------------
class ProgressTracker:
    def __init__(self, config):
        self.config = config
        self.progress_file = os.path.join(config.PREPROCESSED_PATH, "processing_progress.json")
        
    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'train': {'processed': 0, 'total': 0, 'last_index': -1},
            'val': {'processed': 0, 'total': 0, 'last_index': -1},
            'test': {'processed': 0, 'total': 0, 'last_index': -1},
            'started_at': None,
            'last_updated': None,
            'completed': False
        }
    
    def save_progress(self, progress_data):
        progress_data['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def mark_as_completed(self):
        progress = self.load_progress()
        progress['completed'] = True
        progress['completed_at'] = datetime.now().isoformat()
        self.save_progress(progress)
    
    def should_resume(self):
        progress = self.load_progress()
        return not progress['completed']

# ----------------------------------------------------------------------------
# Audio Loader
# ----------------------------------------------------------------------------
class AudioLoader:
    @staticmethod
    def load_audio(filepath, target_sr=16000):
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e1:
            try:
                if filepath.endswith('.flac'):
                    waveform, sr = torchaudio.load(filepath, format='flac')
                elif filepath.endswith('.wav'):
                    waveform, sr = torchaudio.load(filepath, format='wav')
                else:
                    raise e1
            except Exception:
                import soundfile as sf
                waveform_np, sr = sf.read(filepath)
                waveform = torch.from_numpy(waveform_np).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
            
        return waveform, sr

# ----------------------------------------------------------------------------
# Audio Preprocessor
# ----------------------------------------------------------------------------
class AudioPreprocessor:
    def __init__(self, config):
        self.config = config
        self._setup_transforms()
        self.processing_steps = {}
    
    def _setup_transforms(self):
        self.spectrogram = T.Spectrogram(
            n_fft=self.config.N_FFT,
            win_length=self.config.WIN_LENGTH,
            hop_length=self.config.HOP_LENGTH,
            power=None,
            normalized=False
        )
        
        self.mel_scale = T.MelScale(
            n_mels=self.config.N_MELS,
            sample_rate=self.config.SAMPLE_RATE,
            f_min=self.config.F_MIN,
            f_max=self.config.F_MAX,
            n_stft=self.config.N_FFT // 2 + 1
        )
        
        self.window = torch.hann_window(self.config.WIN_LENGTH)
    
    def pre_emphasis(self, waveform, alpha=0.97):
        emphasized = torch.zeros_like(waveform)
        emphasized[:, 0] = waveform[:, 0]
        emphasized[:, 1:] = waveform[:, 1:] - alpha * waveform[:, :-1]
        return emphasized
    
    def compute_log_mel_spectrogram(self, waveform, save_steps=False, sample_id="sample"):
        if save_steps:
            self.processing_steps[sample_id] = {}
        
        if save_steps:
            self.processing_steps[sample_id]['original_waveform'] = waveform.clone()
        
        waveform_pre = self.pre_emphasis(waveform, self.config.PREEMPHASIS)
        if save_steps:
            self.processing_steps[sample_id]['preemphasized_waveform'] = waveform_pre.clone()
        
        if save_steps and waveform_pre.shape[1] > self.config.WIN_LENGTH:
            start_idx = waveform_pre.shape[1] // 2
            frame = waveform_pre[:, start_idx:start_idx + self.config.WIN_LENGTH]
            windowed_frame = frame * self.window
            self.processing_steps[sample_id]['windowed_frame'] = {
                'original': frame,
                'windowed': windowed_frame,
                'window': self.window
            }
        
        complex_spec = self.spectrogram(waveform_pre)
        if save_steps:
            self.processing_steps[sample_id]['complex_spectrogram'] = complex_spec.clone()
        
        magnitude = torch.abs(complex_spec)
        if save_steps:
            self.processing_steps[sample_id]['magnitude_spectrogram'] = magnitude.clone()
        
        power_spec = magnitude ** 2
        if save_steps:
            self.processing_steps[sample_id]['power_spectrogram'] = power_spec.clone()
        
        mel_spec = self.mel_scale(power_spec)
        if save_steps:
            self.processing_steps[sample_id]['mel_spectrogram'] = mel_spec.clone()
        
        log_mel_spec = torch.log(mel_spec + 1e-10)
        if save_steps:
            self.processing_steps[sample_id]['log_mel_spectrogram'] = log_mel_spec.clone()
        
        return log_mel_spec
    
    def save_preprocessed_features(self, raw_features, clean_features, rt60, save_path):
        data_dict = {
            'raw_features': raw_features.numpy(),
            'clean_features': clean_features.numpy(),
            'rt60': rt60,
            'config': {
                'sample_rate': self.config.SAMPLE_RATE,
                'n_fft': self.config.N_FFT,
                'n_mels': self.config.N_MELS,
                'hop_length': self.config.HOP_LENGTH,
                'win_length': self.config.WIN_LENGTH
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)

# ----------------------------------------------------------------------------
# Visualization Functions
# ----------------------------------------------------------------------------
def plot_waveform(waveform, title, ax):
    time_axis = np.arange(len(waveform)) / Config.SAMPLE_RATE
    ax.plot(time_axis, waveform, 'b-', linewidth=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, time_axis[-1]])

def plot_windowed_frame(frame_data, title, ax):
    original = frame_data['original'][0].numpy()
    windowed = frame_data['windowed'][0].numpy()
    window = frame_data['window'].numpy()
    
    time_axis = np.arange(len(original)) / Config.SAMPLE_RATE * 1000
    
    ax.plot(time_axis, original, 'b-', label='Original Frame', linewidth=1, alpha=0.7)
    ax.plot(time_axis, windowed, 'r-', label='Windowed Frame', linewidth=1.5)
    ax.plot(time_axis, window * np.max(original), 'g--', label='Window Function', linewidth=1, alpha=0.5)
    
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

def plot_spectrogram(spec, title, ax, cmap, symmetric=False):
    if symmetric:
        vmax = np.max(np.abs(spec))
        vmin = -vmax
        im = ax.imshow(spec, aspect='auto', origin='lower', 
                      cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap=cmap)
    
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Frequency Bin')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_statistical_comparison(steps_raw, steps_clean, ax):
    raw_logmel = steps_raw['log_mel_spectrogram'][0].numpy()
    clean_logmel = steps_clean['log_mel_spectrogram'][0].numpy()
    
    raw_mean = raw_logmel.mean(axis=1)
    clean_mean = clean_logmel.mean(axis=1)
    raw_std = raw_logmel.std(axis=1)
    clean_std = clean_logmel.std(axis=1)
    
    mel_bands = np.arange(len(raw_mean))
    
    ax.fill_between(mel_bands, raw_mean - raw_std, raw_mean + raw_std, 
                   alpha=0.3, color='red', label='Raw ±1σ')
    ax.plot(mel_bands, raw_mean, 'r-', linewidth=2, label='Raw Mean')
    
    ax.fill_between(mel_bands, clean_mean - clean_std, clean_mean + clean_std, 
                   alpha=0.3, color='blue', label='Clean ±1σ')
    ax.plot(mel_bands, clean_mean, 'b-', linewidth=2, label='Clean Mean')
    
    ax.set_title('Statistical Comparison (Mean ± Std)', fontsize=9)
    ax.set_xlabel('Mel Band Index')
    ax.set_ylabel('Log Magnitude')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

def display_sample_visualization(steps_raw, steps_clean, sample_id, rt60):
    fig = plt.figure(figsize=(20, 12))
    
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.4, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_waveform(steps_raw['original_waveform'][0].numpy(), 
                 "Raw: Original Waveform", ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_waveform(steps_raw['preemphasized_waveform'][0].numpy(), 
                 "Raw: After Pre-emphasis", ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_waveform(steps_clean['original_waveform'][0].numpy(), 
                 "Clean: Original Waveform", ax3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    plot_waveform(steps_clean['preemphasized_waveform'][0].numpy(), 
                 "Clean: After Pre-emphasis", ax4)
    
    if 'windowed_frame' in steps_raw:
        ax5 = fig.add_subplot(gs[0, 4])
        plot_windowed_frame(steps_raw['windowed_frame'], "Raw: Windowing", ax5)
    
    if 'windowed_frame' in steps_clean:
        ax6 = fig.add_subplot(gs[0, 5])
        plot_windowed_frame(steps_clean['windowed_frame'], "Clean: Windowing", ax6)
    
    ax7 = fig.add_subplot(gs[1, 0])
    plot_spectrogram(steps_raw['magnitude_spectrogram'][0].numpy(),
                    "Raw: Magnitude Spectrogram", ax7, 'magma')
    
    ax8 = fig.add_subplot(gs[1, 1])
    plot_spectrogram(steps_clean['magnitude_spectrogram'][0].numpy(),
                    "Clean: Magnitude Spectrogram", ax8, 'magma')
    
    ax9 = fig.add_subplot(gs[1, 2])
    plot_spectrogram(steps_raw['power_spectrogram'][0].numpy(),
                    "Raw: Power Spectrogram", ax9, 'hot')
    
    ax10 = fig.add_subplot(gs[1, 3])
    plot_spectrogram(steps_clean['power_spectrogram'][0].numpy(),
                    "Clean: Power Spectrogram", ax10, 'hot')
    
    ax11 = fig.add_subplot(gs[2, 0])
    plot_spectrogram(steps_raw['mel_spectrogram'][0].numpy(),
                    "Raw: Mel Spectrogram", ax11, 'inferno')
    
    ax12 = fig.add_subplot(gs[2, 1])
    plot_spectrogram(steps_clean['mel_spectrogram'][0].numpy(),
                    "Clean: Mel Spectrogram", ax12, 'inferno')
    
    ax13 = fig.add_subplot(gs[2, 2])
    plot_spectrogram(steps_raw['log_mel_spectrogram'][0].numpy(),
                    "Raw: Log-Mel (Final Feature)", ax13, 'viridis')
    
    ax14 = fig.add_subplot(gs[2, 3])
    plot_spectrogram(steps_clean['log_mel_spectrogram'][0].numpy(),
                    "Clean: Log-Mel (Final Feature)", ax14, 'viridis')
    
    ax15 = fig.add_subplot(gs[2, 4])
    diff = steps_raw['log_mel_spectrogram'][0].numpy() - steps_clean['log_mel_spectrogram'][0].numpy()
    plot_spectrogram(diff, "Difference: Reverberation", ax15, 'RdBu_r', symmetric=True)
    
    ax16 = fig.add_subplot(gs[2, 5])
    plot_statistical_comparison(steps_raw, steps_clean, ax16)
    
    fig.suptitle(f'Sample {sample_id} - RT60: {rt60:.2f}s - Preprocessing Pipeline', 
                fontsize=12, y=0.98)
    plt.show()
    
    return fig

# ----------------------------------------------------------------------------
# Main Preprocessing Pipeline with Resume Capability
# ----------------------------------------------------------------------------
def preprocess_full_dataset(resume=False):
    print("\nStarting full dataset preprocessing")
    if resume:
        print("Resuming from previous progress")
    print("-----------------------------------")
    
    config = Config()
    progress_tracker = ProgressTracker(config)
    
    print("Loading dataset mapping...")
    df = pd.read_csv(config.MAPPING_PATH)
    df = df[df['status'] == 'success']
    total_samples = len(df)
    print(f"Found {total_samples} valid samples")
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    test_split = int(total_samples * config.TEST_SPLIT)
    val_split = int(total_samples * config.VAL_SPLIT)
    
    test_df = df[:test_split]
    val_df = df[test_split:test_split + val_split]
    train_df = df[test_split + val_split:]
    
    print("\nDataset splits created")
    print("----------------------")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    audio_loader = AudioLoader()
    processed_samples = 0
    failed_samples = 0
    visualization_data = []
    
    # CSV to track saved files
    csv_save_path = os.path.join(config.PREPROCESSED_PATH, "preprocessed_mapping.csv")
    
    # Load existing mapping if resuming
    if resume and os.path.exists(csv_save_path):
        preprocessed_df = pd.read_csv(csv_save_path)
        existing_files = set(preprocessed_df['preprocessed_file'].tolist())
        print(f"\nFound {len(existing_files)} previously processed files")
    else:
        preprocessed_info = []
        existing_files = set()
    
    # Load progress if resuming
    if resume:
        progress = progress_tracker.load_progress()
        print(f"\nPrevious progress found:")
        print(f"  Train: {progress['train']['processed']}/{progress['train']['total']} processed")
        print(f"  Val: {progress['val']['processed']}/{progress['val']['total']} processed")
        print(f"  Test: {progress['test']['processed']}/{progress['test']['total']} processed")
        
        resume_choice = input("\nResume from previous progress? (y/n): ").lower().strip()
        if resume_choice != 'y':
            resume = False
            progress = {
                'train': {'processed': 0, 'total': len(train_df), 'last_index': -1},
                'val': {'processed': 0, 'total': len(val_df), 'last_index': -1},
                'test': {'processed': 0, 'total': len(test_df), 'last_index': -1},
                'started_at': datetime.now().isoformat(),
                'last_updated': None,
                'completed': False
            }
    else:
        progress = {
            'train': {'processed': 0, 'total': len(train_df), 'last_index': -1},
            'val': {'processed': 0, 'total': len(val_df), 'last_index': -1},
            'test': {'processed': 0, 'total': len(test_df), 'last_index': -1},
            'started_at': datetime.now().isoformat(),
            'last_updated': None,
            'completed': False
        }
    
    progress_tracker.save_progress(progress)
    
    print("\nProcessing dataset")
    print("------------------")
    print("This will process all audio files and save preprocessed features")
    print(f"Saved to: {config.PREPROCESSED_PATH}")
    
    datasets = [
        ('train', train_df, progress['train']),
        ('val', val_df, progress['val']),
        ('test', test_df, progress['test'])
    ]
    
    for dataset_name, dataset_df, dataset_progress in datasets:
        dataset_save_dir = os.path.join(config.PREPROCESSED_PATH, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        
        total_to_process = len(dataset_df)
        start_idx = dataset_progress['last_index'] + 1 if resume else 0
        
        if start_idx >= total_to_process:
            print(f"\n{dataset_name.capitalize()} set already fully processed")
            continue
        
        print(f"\nProcessing {dataset_name} set ({total_to_process} samples)")
        print(f"Resuming from index {start_idx}")
        print(f"Saving to: {dataset_save_dir}")
        
        with tqdm(total=total_to_process, initial=start_idx, desc=f"{dataset_name}", unit="samples") as pbar:
            for idx in range(start_idx, len(dataset_df)):
                row = dataset_df.iloc[idx]
                try:
                    raw_path = row['output_path']
                    clean_path = row['clean_path']
                    raw_filename = os.path.basename(raw_path)
                    clean_filename = os.path.basename(clean_path)
                    
                    # Check if already processed
                    base_name = os.path.splitext(raw_filename)[0]
                    save_filename = f"{base_name}.pkl"
                    
                    if resume and save_filename in existing_files:
                        pbar.update(1)
                        dataset_progress['processed'] += 1
                        dataset_progress['last_index'] = idx
                        continue
                    
                    raw_wave, _ = audio_loader.load_audio(raw_path, config.SAMPLE_RATE)
                    clean_wave, _ = audio_loader.load_audio(clean_path, config.SAMPLE_RATE)
                    
                    min_len = min(raw_wave.shape[1], clean_wave.shape[1])
                    raw_wave = raw_wave[:, :min_len]
                    clean_wave = clean_wave[:, :min_len]
                    
                    preprocessor = AudioPreprocessor(config)
                    
                    # Process and save features
                    raw_features = preprocessor.compute_log_mel_spectrogram(raw_wave)
                    clean_features = preprocessor.compute_log_mel_spectrogram(clean_wave)
                    
                    min_frames = min(raw_features.shape[2], clean_features.shape[2])
                    raw_features = raw_features[:, :, :min_frames]
                    clean_features = clean_features[:, :, :min_frames]
                    
                    # Save preprocessed features
                    save_path = os.path.join(dataset_save_dir, save_filename)
                    
                    preprocessor.save_preprocessed_features(
                        raw_features, clean_features, row['rt60_value'], save_path
                    )
                    
                    # Track in CSV
                    if not resume or save_filename not in existing_files:
                        preprocessed_info.append({
                            'dataset': dataset_name,
                            'original_raw': raw_filename,
                            'original_clean': clean_filename,
                            'preprocessed_file': save_filename,
                            'save_path': save_path,
                            'rt60': row['rt60_value'],
                            'raw_shape': str(raw_features.shape),
                            'clean_shape': str(clean_features.shape)
                        })
                    
                    processed_samples += 1
                    dataset_progress['processed'] += 1
                    dataset_progress['last_index'] = idx
                    
                    # Save progress every 100 samples
                    if idx % 100 == 0:
                        progress_tracker.save_progress(progress)
                    
                    # Capture visualization data for test samples
                    if idx < config.VISUALIZATION_SAMPLES and dataset_name == 'test':
                        preprocessor_vis = AudioPreprocessor(config)
                        raw_features_vis = preprocessor_vis.compute_log_mel_spectrogram(
                            raw_wave, save_steps=True, sample_id="raw"
                        )
                        clean_features_vis = preprocessor_vis.compute_log_mel_spectrogram(
                            clean_wave, save_steps=True, sample_id="clean"
                        )
                        
                        visualization_data.append({
                            'raw_steps': preprocessor_vis.processing_steps['raw'],
                            'clean_steps': preprocessor_vis.processing_steps['clean'],
                            'sample_id': idx + 1,
                            'rt60': row['rt60_value'],
                            'raw_path': raw_path,
                            'clean_path': clean_path,
                            'preprocessed_path': save_path
                        })
                        
                except Exception as e:
                    failed_samples += 1
                    pbar.set_postfix({'failed': failed_samples})
                    continue
                
                pbar.update(1)
                pbar.set_postfix({
                    'processed': processed_samples,
                    'failed': failed_samples,
                    'resume_idx': idx
                })
        
        # Update progress after each dataset
        progress_tracker.save_progress(progress)
    
    # Save CSV with mapping information
    preprocessed_df = pd.DataFrame(preprocessed_info)
    if resume and os.path.exists(csv_save_path):
        # Append to existing CSV
        existing_df = pd.read_csv(csv_save_path)
        combined_df = pd.concat([existing_df, preprocessed_df], ignore_index=True)
        combined_df.to_csv(csv_save_path, index=False)
    else:
        preprocessed_df.to_csv(csv_save_path, index=False)
    
    # Mark as completed
    progress_tracker.mark_as_completed()
    
    print("\nProcessing complete")
    print("-------------------")
    print(f"Successfully processed: {processed_samples} samples")
    print(f"Failed to process: {failed_samples} samples")
    print(f"Preprocessed features saved to: {config.PREPROCESSED_PATH}")
    print(f"Mapping CSV saved to: {csv_save_path}")
    print(f"Progress tracking saved to: {progress_tracker.progress_file}")
    
    # Save dataset info
    dataset_info = {
        'total_samples': processed_samples,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'sample_rate': config.SAMPLE_RATE,
        'feature_dim': config.N_MELS,
        'preprocessed_dir': config.PREPROCESSED_PATH,
        'created_at': datetime.now().isoformat(),
        'resumed': resume
    }
    
    info_path = os.path.join(config.PREPROCESSED_PATH, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset information saved to: {info_path}")
    
    return train_df, val_df, test_df, visualization_data, preprocessed_df

# ----------------------------------------------------------------------------
# Display Visualizations
# ----------------------------------------------------------------------------
def display_visualizations(visualization_data):
    if not visualization_data:
        print("No visualization data available")
        return
    
    print(f"\nDisplaying preprocessing visualizations for {len(visualization_data)} test samples")
    print("------------------------------------------------------------------------------------")
    
    for i, vis_data in enumerate(visualization_data):
        print(f"\nVisualization {i+1}: Sample {vis_data['sample_id']}")
        print(f"  Raw file: {os.path.basename(vis_data['raw_path'])}")
        print(f"  Clean file: {os.path.basename(vis_data['clean_path'])}")
        print(f"  RT60: {vis_data['rt60']:.2f}s")
        print(f"  Preprocessed saved to: {vis_data['preprocessed_path']}")
        
        fig = display_sample_visualization(
            vis_data['raw_steps'],
            vis_data['clean_steps'],
            vis_data['sample_id'],
            vis_data['rt60']
        )
        
        save_path = os.path.join(Config.VISUALIZATION_DIR, 
                                f"pipeline_visualization_sample_{vis_data['sample_id']}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Visualization saved to: {save_path}")

# ----------------------------------------------------------------------------
# Check if should resume
# ----------------------------------------------------------------------------
def should_resume_processing():
    config = Config()
    progress_tracker = ProgressTracker(config)
    
    if not progress_tracker.should_resume():
        return False
    
    progress = progress_tracker.load_progress()
    
    print("\nPrevious processing detected")
    print("---------------------------")
    print(f"Started at: {progress.get('started_at', 'Unknown')}")
    print(f"Last updated: {progress.get('last_updated', 'Never')}")
    print(f"Completed: {progress.get('completed', False)}")
    print()
    print("Progress summary:")
    print(f"  Training: {progress['train']['processed']}/{progress['train']['total']} ({progress['train']['processed']/max(progress['train']['total'], 1)*100:.1f}%)")
    print(f"  Validation: {progress['val']['processed']}/{progress['val']['total']} ({progress['val']['processed']/max(progress['val']['total'], 1)*100:.1f}%)")
    print(f"  Test: {progress['test']['processed']}/{progress['test']['total']} ({progress['test']['processed']/max(progress['test']['total'], 1)*100:.1f}%)")
    
    resume = input("\nResume processing from where it left off? (y/n): ").lower().strip()
    return resume == 'y'

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    print("Speech Dereverberation Preprocessing Pipeline")
    print("=============================================")
    
    # Check if we should resume
    resume = should_resume_processing()
    
    if resume:
        print("\nResuming processing...")
    else:
        print("\nStarting fresh processing...")
    
    print("Processing all audio files and saving preprocessed features\n")
    
    train_df, val_df, test_df, visualization_data, preprocessed_df = preprocess_full_dataset(resume=resume)
    
    elapsed_time = time.time() - start_time
    print(f"\nPreprocessing summary")
    print("=====================")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Visualizations created: {len(visualization_data)}")
    print(f"Resume capability: Enabled")
    print(f"\nPreprocessed files saved in:")
    print(f"  {Config.PREPROCESSED_PATH}/train/")
    print(f"  {Config.PREPROCESSED_PATH}/val/")
    print(f"  {Config.PREPROCESSED_PATH}/test/")
    print(f"\nMapping CSV: {Config.PREPROCESSED_PATH}/preprocessed_mapping.csv")
    print(f"Progress tracking: {Config.PREPROCESSED_PATH}/processing_progress.json")
    
    if visualization_data:
        display_visualizations(visualization_data)
    
    print("\nPreprocessing complete. Ready for model training.")