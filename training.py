import os
import sys
import time
import pickle
import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
 
# Configuration 
class Config:
    """Optimized configuration for 4-hour training."""
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "FinalDataset" / "preProcessedSpeech"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    RESULTS_DIR = BASE_DIR / "results"
    STATS_FILE = BASE_DIR / "data_stats.json"
    
    # Data parameters
    INPUT_MELS = 80
    DATA_MEAN = -7.1736
    DATA_STD = 4.1806
    MAX_SEQ_LEN = 1000
    
    # Model architecture
    D_MODEL = 256
    N_LAYERS = 6
    N_HEADS = 8
    CONV_KERNEL_SIZE = 31
    FFN_EXPANSION = 4
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    TOTAL_STEPS = 50000
    GRAD_CLIP = 1.0
    
    # Checkpointing
    CHECKPOINT_INTERVAL_MIN = 30
    VAL_INTERVAL_STEPS = 500
    PLOT_INTERVAL_STEPS = 1000
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0
    
    def __init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        
        print("Training Configuration:")
        print(f"  Data stats: mean={self.DATA_MEAN:.4f}, std={self.DATA_STD:.4f}")
        print(f"  Max sequence length: {self.MAX_SEQ_LEN}")
        print(f"  Device: {self.DEVICE}")
        if self.DEVICE == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
        print(f"  Batch size: {self.BATCH_SIZE}")
        print(f"  Total steps planned: {self.TOTAL_STEPS}")
        print(f"  Checkpoint interval: {self.CHECKPOINT_INTERVAL_MIN} minutes")
        print()
 
# Data Loading 
class FixedLengthLogMelDataset(Dataset):
    """Dataset with fixed maximum length."""
    
    def __init__(self, data_dir, split='train', max_samples=None, config=None):
        self.data_dir = Path(data_dir) / split
        self.config = config
        self.max_seq_len = config.MAX_SEQ_LEN
        
        if not self.data_dir.exists():
            raise ValueError(f"Directory {self.data_dir} does not exist")
        
        self.files = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.pkl'])
        
        if max_samples:
            self.files = self.files[:max_samples]
        
        print(f"Loaded {len(self.files)} files from {split} set")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Load data
            x_reverb = torch.tensor(data['raw_features'], dtype=torch.float32)
            x_clean = torch.tensor(data['clean_features'], dtype=torch.float32)
            rt60 = torch.tensor(data['rt60'], dtype=torch.float32)
            
            # Ensure proper shape
            if x_reverb.dim() == 2:
                x_reverb = x_reverb.unsqueeze(0)
            if x_clean.dim() == 2:
                x_clean = x_clean.unsqueeze(0)
            
            # Get sequence length
            seq_len = x_reverb.shape[2]
            
            # Handle long sequences
            if seq_len > self.max_seq_len:
                start = (seq_len - self.max_seq_len) // 2
                x_reverb = x_reverb[:, :, start:start + self.max_seq_len]
                x_clean = x_clean[:, :, start:start + self.max_seq_len]
                seq_len = self.max_seq_len
            
            # Normalize
            x_reverb = (x_reverb - self.config.DATA_MEAN) / self.config.DATA_STD
            x_clean = (x_clean - self.config.DATA_MEAN) / self.config.DATA_STD
            
            # Add RT60 channel
            rt60_channel = rt60.repeat(1, 1, seq_len)
            x_input = torch.cat([x_reverb, rt60_channel], dim=1)
            
            return {
                'input': x_input.squeeze(0),
                'target': x_clean.squeeze(0),
                'rt60': rt60,
                'seq_len': seq_len,
                'file': file_path.name
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)[:100]}")
            return self.__getitem__((idx + 1) % len(self))

def efficient_collate(batch):
    """Efficient collate function."""
    batch_size = len(batch)
    
    # Find max length in this batch
    max_len = min(max(item['input'].shape[1] for item in batch), 1000)
    
    # Pre-allocate tensors
    inputs = torch.zeros(batch_size, 81, max_len)
    targets = torch.zeros(batch_size, 80, max_len)
    rt60s = torch.zeros(batch_size)
    seq_lens = []
    
    for i, item in enumerate(batch):
        seq_len = min(item['input'].shape[1], max_len)
        inputs[i, :, :seq_len] = item['input'][:, :seq_len]
        targets[i, :, :seq_len] = item['target'][:, :seq_len]
        rt60s[i] = item['rt60']
        seq_lens.append(seq_len)
    
    return {
        'input': inputs,
        'target': targets,
        'rt60': rt60s,
        'seq_len': torch.tensor(seq_lens),
        'file': [item['file'] for item in batch]
    }
 
# Conformer Model 
class SimpleAttention(nn.Module):
    """Simplified attention."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        
        return out

class SimpleConformerBlock(nn.Module):
    """Simplified Conformer block."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        # Attention part
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = SimpleAttention(d_model, n_heads, dropout)
        
        # Conv part
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # FFN part
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Attention with residual
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, mask)
        x = residual + x
        
        # Conv with residual
        residual = x
        x = self.conv_norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = residual + x
        
        # FFN with residual
        x = x + self.ffn(x)
        
        return x

class FastConformer(nn.Module):
    """Fast Conformer model."""
    
    def __init__(self, config):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv1d(81, config.D_MODEL, kernel_size=3, padding=1)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, config.D_MODEL) * 0.02)
        
        # Conformer blocks
        self.blocks = nn.ModuleList([
            SimpleConformerBlock(config.D_MODEL, config.N_HEADS, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])
        
        # Output
        self.output_proj = nn.Conv1d(config.D_MODEL, config.INPUT_MELS, kernel_size=1)
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, seq_lens=None):
        # x: [B, 81, T]
        B, _, T = x.shape
        
        # Input projection
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        if T <= 1000:
            x = x + self.pos_embed[:, :T, :]
        else:
            x = x[:, :1000, :] + self.pos_embed
        
        # Conformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = x.transpose(1, 2)
        x = self.output_proj(x)
        
        return x
 
# Training System with Progress Bars 
class TrainerWithProgress:
    """Training system with progress bars and graphing."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Data
        print("Loading datasets...")
        self.train_dataset = FixedLengthLogMelDataset(
            config.DATA_DIR, 'train', max_samples=None, config=config
        )
        self.val_dataset = FixedLengthLogMelDataset(
            config.DATA_DIR, 'val', max_samples=500, config=config
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=efficient_collate,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=efficient_collate,
            pin_memory=True
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print()
        
        # Model
        print("Creating model...")
        self.model = FastConformer(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer and scheduler
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.TOTAL_STEPS)
        
        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'steps': [],
            'wall_time': [],
            'learning_rate': [],
            'checkpoints': []
        }
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        self.last_checkpoint_time = time.time()
        self.last_plot_time = time.time()
        
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        # Move data
        inputs = batch['input'].to(self.device)
        targets = batch['target'].to(self.device)
        seq_lens = batch['seq_len'].to(self.device)
        
        # Forward
        self.optimizer.zero_grad()
        outputs = self.model(inputs, seq_lens)
        
        # Loss
        loss = 0
        valid_samples = 0
        for i, seq_len in enumerate(seq_lens):
            if seq_len > 0:
                loss += self.criterion(outputs[i, :, :seq_len], targets[i, :, :seq_len])
                valid_samples += 1
        
        if valid_samples > 0:
            loss = loss / valid_samples
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            return loss.item()
        
        return 0.0
    
    def validate(self):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            for batch in val_pbar:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                seq_lens = batch['seq_len'].to(self.device)
                
                outputs = self.model(inputs, seq_lens)
                
                for i, seq_len in enumerate(seq_lens):
                    if seq_len > 0:
                        loss = self.criterion(outputs[i, :, :seq_len], targets[i, :, :seq_len])
                        total_loss += loss.item()
                        total_samples += 1
        
        return total_loss / total_samples if total_samples > 0 else float('inf')
    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': vars(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            filename = f"conformer_best_step{self.global_step}_{timestamp}.pt"
        else:
            filename = f"conformer_step{self.global_step}_{timestamp}.pt"
        
        filepath = self.config.CHECKPOINT_DIR / filename
        torch.save(checkpoint, filepath)
        
        # Record
        self.history['checkpoints'].append({
            'step': self.global_step,
            'file': filename,
            'is_best': is_best,
            'timestamp': checkpoint['timestamp']
        })
        
        return filepath
    
    def plot_progress(self):
        """Plot training progress."""
        if len(self.history['steps']) < 2:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss vs Steps
        ax = axes[0, 0]
        steps = self.history['steps']
        train_loss = self.history['train_loss']
        
        # Plot every 10th point for clarity if too many points
        if len(steps) > 1000:
            indices = np.linspace(0, len(steps)-1, 100, dtype=int)
            steps_plot = [steps[i] for i in indices]
            train_loss_plot = [train_loss[i] for i in indices]
            ax.plot(steps_plot, train_loss_plot, 'b-', alpha=0.6, linewidth=0.5)
        else:
            ax.plot(steps, train_loss, 'b-', alpha=0.6, linewidth=0.5)
        
        # Plot validation loss
        if self.history['val_loss']:
            val_steps = self.history['steps'][:len(self.history['val_loss']) * self.config.VAL_INTERVAL_STEPS:self.config.VAL_INTERVAL_STEPS]
            val_steps = val_steps[:len(self.history['val_loss'])]
            ax.plot(val_steps, self.history['val_loss'], 'r-', linewidth=2, label='Validation')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss vs Wall Time
        ax = axes[0, 1]
        wall_time = self.history['wall_time']
        ax.plot(wall_time, train_loss, 'b-', alpha=0.6, linewidth=0.5)
        if self.history['val_loss']:
            val_times = wall_time[:len(self.history['val_loss']) * self.config.VAL_INTERVAL_STEPS:self.config.VAL_INTERVAL_STEPS]
            val_times = val_times[:len(self.history['val_loss'])]
            ax.plot(val_times, self.history['val_loss'], 'r-', linewidth=2)
        ax.set_xlabel('Wall Time (hours)')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Loss vs Training Time')
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 0]
        if self.history['learning_rate']:
            lr_steps = steps[:len(self.history['learning_rate'])]
            ax.plot(lr_steps, self.history['learning_rate'], 'g-', linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        # Recent Loss Distribution
        ax = axes[1, 1]
        if len(train_loss) > 100:
            recent_losses = train_loss[-100:]
            ax.hist(recent_losses, bins=20, alpha=0.7, color='blue')
            ax.axvline(np.mean(recent_losses), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(recent_losses):.4f}')
            ax.set_xlabel('Loss Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Recent Training Loss Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.config.RESULTS_DIR / f"training_progress_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def save_history(self):
        """Save training history to JSON."""
        # Convert to serializable format
        serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                serializable[key] = [float(v) if isinstance(v, (torch.Tensor, np.generic)) else v for v in value]
            else:
                serializable[key] = value
        
        filepath = self.config.RESULTS_DIR / "training_history.json"
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        return filepath
    
    def run(self, total_hours=4):
        """Main training loop with progress bars."""
        print(f"\nStarting training for {total_hours} hours")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Checkpoint every {self.config.CHECKPOINT_INTERVAL_MIN} minutes")
        print(f"Validation every {self.config.VAL_INTERVAL_STEPS} steps")
        print()
        
        total_seconds = total_hours * 3600
        total_steps_estimate = min(self.config.TOTAL_STEPS, 20000)  # Estimate for progress bar
        
        # Main progress bar
        main_pbar = tqdm(
            total=total_steps_estimate,
            desc="Training Progress",
            unit="step",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]'
        )
        
        # Time progress bar
        time_pbar = tqdm(
            total=total_seconds,
            desc="Time Progress",
            unit="s",
            position=1,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]'
        )
        
        try:
            while True:
                # Check time budget
                elapsed = time.time() - self.start_time
                time_pbar.update(int(elapsed) - time_pbar.n)
                
                if elapsed >= total_seconds:
                    print(f"\nTime budget reached: {elapsed/3600:.2f} hours")
                    break
                
                if self.global_step >= self.config.TOTAL_STEPS:
                    print(f"\nMaximum steps reached: {self.global_step}")
                    break
                
                # Training loop over batches
                batch_pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch",
                    leave=False,
                    position=2,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches'
                )
                
                for batch_idx, batch in enumerate(batch_pbar):
                    # Time check
                    elapsed = time.time() - self.start_time
                    if elapsed >= total_seconds:
                        break
                    
                    # Training step
                    loss = self.train_step(batch)
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    # Record history
                    self.history['train_loss'].append(loss)
                    self.history['steps'].append(self.global_step)
                    self.history['wall_time'].append(elapsed / 3600)
                    self.history['learning_rate'].append(current_lr)
                    
                    # Update progress bars
                    main_pbar.update(1)
                    main_pbar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{current_lr:.2e}'})
                    
                    # Validation
                    if self.global_step % self.config.VAL_INTERVAL_STEPS == 0:
                        val_loss = self.validate()
                        self.history['val_loss'].append(val_loss)
                        
                        # Update best loss
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(is_best=True)
                            print(f"\n[Step {self.global_step}] New best validation loss: {val_loss:.6f}")
                    
                    # Checkpointing by time
                    current_time = time.time()
                    if current_time - self.last_checkpoint_time >= self.config.CHECKPOINT_INTERVAL_MIN * 60:
                        checkpoint_path = self.save_checkpoint()
                        plot_path = self.plot_progress()
                        
                        time_str = datetime.now().strftime('%H:%M:%S')
                        print(f"\n[{time_str}] Checkpoint saved at step {self.global_step}")
                        print(f"  Training loss: {loss:.6f}")
                        print(f"  Learning rate: {current_lr:.6f}")
                        print(f"  Saved to: {checkpoint_path.name}")
                        if plot_path:
                            print(f"  Plot saved: {plot_path.name}")
                        
                        self.last_checkpoint_time = current_time
                    
                    # Plot periodically
                    if self.global_step % self.config.PLOT_INTERVAL_STEPS == 0:
                        self.plot_progress()
                    
                    self.global_step += 1
                    
                    # Update time progress
                    time_pbar.update(int(time.time() - self.start_time) - time_pbar.n)
                
                batch_pbar.close()
                
                # Save history periodically
                if self.global_step % 1000 == 0:
                    self.save_history()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Close progress bars
            main_pbar.close()
            time_pbar.close()
            
            # Final save
            
            print("Training completed!")
            
            final_checkpoint = self.save_checkpoint()
            final_plot = self.plot_progress()
            history_file = self.save_history()
            
            print(f"\nFinal results:")
            print(f"  Total steps: {self.global_step}")
            print(f"  Best validation loss: {self.best_val_loss:.6f}")
            print(f"  Final training loss: {self.history['train_loss'][-1] if self.history['train_loss'] else 'N/A':.6f}")
            print(f"  Total training time: {elapsed/3600:.2f} hours")
            print(f"\nFiles saved:")
            print(f"  Final checkpoint: {final_checkpoint}")
            print(f"  Training plots: {final_plot}")
            print(f"  History file: {history_file}")
        
        return self.history
 
# Main Execution 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=4.0, help='Training hours')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--test', action='store_true', help='Test mode only')
    parser.add_argument('--max-train', type=int, default=None, help='Max training samples')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.BATCH_SIZE = args.batch_size
    
    if args.test:
        print("Running in test mode...")
        
        # Quick test
        dataset = FixedLengthLogMelDataset(config.DATA_DIR, 'train', max_samples=10, config=config)
        loader = DataLoader(dataset, batch_size=2, collate_fn=efficient_collate)
        batch = next(iter(loader))
        
        model = FastConformer(config).to(config.DEVICE)
        with torch.no_grad():
            output = model(batch['input'].to(config.DEVICE))
        
        print(f"\nTest results:")
        print(f"  Batch input shape: {batch['input'].shape}")
        print(f"  Model output shape: {output.shape}")
        print(f"  Sequence lengths: {batch['seq_len'].tolist()}")
        print("\nTest passed! Ready for training.")
        return
    
    # Full training
    trainer = TrainerWithProgress(config)
    
    # Use max samples if specified
    if args.max_train:
        trainer.train_dataset = FixedLengthLogMelDataset(
            config.DATA_DIR, 'train', max_samples=args.max_train, config=config
        )
        trainer.train_loader = DataLoader(
            trainer.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=efficient_collate,
            pin_memory=True
        )
        print(f"Using {args.max_train} training samples for faster testing")
    
    # Run training
    history = trainer.run(total_hours=args.hours)
    
    
    print("Training summary complete!")

if __name__ == "__main__":

    main()
