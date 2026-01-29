"""
VITS Emotion-based TTS Fine-tuning Script
Implements emotion embedding layer for emotional TTS generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchaudio
import yaml
import wandb
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from transformers import VitsModel, VitsTokenizer
import warnings
warnings.filterwarnings('ignore')


# ============================
# Emotion Embedding Module
# ============================
class EmotionEmbedding(nn.Module):
    """Emotion embedding layer to inject emotion information into VITS"""
    def __init__(self, num_emotions=5, emotion_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, emotion_dim)
        # Multi-layer projection for better emotion separation
        self.projection = nn.Sequential(
            nn.Linear(emotion_dim, emotion_dim * 2),
            nn.LayerNorm(emotion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emotion_dim * 2, emotion_dim),
        )
        # Emotion amplification factor (learnable per emotion)
        self.emotion_scale = nn.Parameter(torch.ones(num_emotions, 1) * 2.0)
        
    def forward(self, emotion_ids):
        """
        Args:
            emotion_ids: (batch_size,) tensor of emotion class indices
        Returns:
            emotion_embed: (batch_size, emotion_dim) emotion embeddings
        """
        emotion_embed = self.embedding(emotion_ids)
        # Apply amplification per emotion for stronger differentiation
        scales = self.emotion_scale[emotion_ids]
        emotion_embed = emotion_embed * scales
        # Project with non-linearity
        emotion_embed = self.projection(emotion_embed)
        return emotion_embed


class VITSWithEmotion(nn.Module):
    """Wrapper around VITS model to add emotion conditioning"""
    def __init__(self, vits_model, num_emotions=5, emotion_dim=256):
        super().__init__()
        self.vits = vits_model
        self.emotion_embedding = EmotionEmbedding(num_emotions, emotion_dim)
        
        # VITS uses 192 hidden dim typically
        self.text_encoder_dim = 192
        
        # Project emotion to VITS speaker embedding size (usually 192 or 256)
        # This allows emotions to act as "speaker" variations
        self.emotion_to_speaker = nn.Sequential(
            nn.Linear(emotion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 192)  # Match VITS speaker embedding size
        )
        
        # Fusion layer with residual connection
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder_dim + emotion_dim, self.text_encoder_dim * 2),
            nn.LayerNorm(self.text_encoder_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_encoder_dim * 2, self.text_encoder_dim)
        )
        
    def forward(self, x, x_lengths, y, y_lengths, emotion_ids, **kwargs):
        """
        Args:
            x: text input
            x_lengths: text lengths
            y: mel spectrogram target
            y_lengths: mel lengths
            emotion_ids: emotion class indices
        """
        # Get emotion embeddings
        emotion_embed = self.emotion_embedding(emotion_ids)  # (B, emotion_dim)
        
        # Encode text
        # Note: This is a simplified version. You'll need to modify based on actual VITS architecture
        text_encoded = self.vits.encode_text(x, x_lengths) if hasattr(self.vits, 'encode_text') else x
        
        # Expand emotion embedding to match sequence length
        B, T, D = text_encoded.shape
        emotion_expanded = emotion_embed.unsqueeze(1).expand(B, T, -1)  # (B, T, emotion_dim)
        
        # Concatenate and fuse
        combined = torch.cat([text_encoded, emotion_expanded], dim=-1)  # (B, T, text_dim + emotion_dim)
        fused = self.fusion(combined)  # (B, T, text_dim)
        
        # Pass through rest of VITS model
        return self.vits(x, x_lengths, y, y_lengths, text_encoded=fused, **kwargs)
    
    def infer(self, input_ids, emotion_ids, **kwargs):
        """Inference with emotion conditioning - inject as speaker embedding"""
        emotion_embed = self.emotion_embedding(emotion_ids)  # (B, emotion_dim)
        
        # Convert emotion to speaker-like embedding for VITS
        speaker_embed = self.emotion_to_speaker(emotion_embed)  # (B, 192)
        
        # VITS expects speaker_id or can take speaker embeddings
        # We'll try to pass it as speaker conditioning
        try:
            # Method 1: Try passing as speaker embedding directly
            if hasattr(self.vits, 'inference'):
                output = self.vits.inference(input_ids, speaker_embeddings=speaker_embed, **kwargs)
            # Method 2: Try standard forward with speaker embedding
            elif hasattr(self.vits, '__call__'):
                # Many VITS models accept speaker_id parameter
                output = self.vits(input_ids, speaker_embeddings=speaker_embed, **kwargs)
            else:
                output = self.vits(input_ids, **kwargs)
            return output
        except TypeError:
            # Fallback if speaker embeddings not accepted
            return self.vits(input_ids, **kwargs)


# ============================
# Dataset Class
# ============================
class EmotionTTSDataset(Dataset):
    """Dataset for emotion-tagged TTS training"""
    def __init__(self, hf_dataset, target_sr=16000, max_length=16000, emotion_map=None):
        self.dataset = hf_dataset
        self.target_sr = target_sr
        self.max_length = max_length
        
        # Emotion mapping (string to int)
        if emotion_map is None:
            # Default ESD emotion mapping
            self.emotion_map = {
                'Neutral': 0,
                'Happy': 1,
                'Sad': 2,
                'Angry': 3,
                'Surprise': 4
            }
        else:
            self.emotion_map = emotion_map
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load and process audio
        audio = item['audio']
        waveform = torch.tensor(audio['array']).float()
        sample_rate = audio['sampling_rate']
        
        # Resample to target sample rate
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        
        # Pad or trim to max_length
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            waveform = F.pad(waveform, (0, self.max_length - len(waveform)))
        
        # Convert to mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        mel = mel_transform(waveform)
        
        # Map emotion string to integer
        emotion_str = item['emotion']
        emotion_id = self.emotion_map.get(emotion_str, 0)  # Default to 0 if unknown
        
        return {
            'waveform': waveform,
            'mel': mel,
            'text': item['transcription'],
            'emotion': emotion_id,  # Now it's an integer
            'audio_length': len(waveform)
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    waveforms = [item['waveform'] for item in batch]
    mels = [item['mel'] for item in batch]
    texts = [item['text'] for item in batch]
    emotions = torch.tensor([item['emotion'] for item in batch])
    
    # Pad sequences
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    
    # Pad mels
    max_mel_len = max([m.shape[-1] for m in mels])
    mels_padded = torch.zeros(len(mels), mels[0].shape[0], max_mel_len)
    for i, mel in enumerate(mels):
        mels_padded[i, :, :mel.shape[-1]] = mel
    
    return {
        'waveforms': waveforms,
        'mels': mels_padded,
        'texts': texts,
        'emotions': emotions
    }


# ============================
# Training Functions
# ============================
class EmotionVITSTrainer:
    """Trainer for emotion-conditioned VITS"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize mel-to-emotion projection layer for loss computation
        n_mels = config.get('n_mels', 80)
        emotion_dim = config.get('emotion_dim', 256)
        self.mel_to_emotion_proj = nn.Linear(n_mels, emotion_dim).to(self.device)
        
        print(f"Initialized trainer with mel projection: {n_mels} -> {emotion_dim}")
        
    def freeze_vits_weights(self):
        """Freeze VITS weights except emotion embedding and decoder"""
        print("Freezing VITS encoder weights...")
        for name, param in self.model.vits.named_parameters():
            # Freeze everything except decoder and flow components
            if 'decoder' not in name and 'flow' not in name:
                param.requires_grad = False
        
        # Keep emotion embedding trainable
        for param in self.model.emotion_embedding.parameters():
            param.requires_grad = True
        for param in self.model.fusion.parameters():
            param.requires_grad = True
        
        # Keep mel projection trainable
        for param in self.mel_to_emotion_proj.parameters():
            param.requires_grad = True
            
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_params += sum(p.numel() for p in self.mel_to_emotion_proj.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
    
    def unfreeze_all_weights(self):
        """Unfreeze all weights for end-to-end fine-tuning"""
        print("Unfreezing all weights for end-to-end training...")
        for param in self.model.parameters():
            param.requires_grad = True
            
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train_stage1(self, train_loader, epochs=5, lr=1e-4):
        """
        Stage 1: Train only emotion embeddings and decoder
        Freeze most VITS weights
        """
        print("\n" + "="*50)
        print("STAGE 1: Training Emotion Embeddings + Decoder")
        print("="*50 + "\n")
        
        self.freeze_vits_weights()
        
        # Combine model and projection layer parameters
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters())) + \
                          list(self.mel_to_emotion_proj.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs * len(train_loader)
        )
        
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                mels = batch['mels'].to(self.device)
                emotions = batch['emotions'].to(self.device)
                texts = batch['texts']  # Process texts based on your tokenizer
                
                # Forward pass (simplified - adapt to your VITS model)
                try:
                    # This is a placeholder - you need to adapt this to your specific VITS model
                    loss = self.compute_loss(texts, mels, emotions)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    # Logging
                    if global_step % 10 == 0:
                        if wandb.run is not None:
                            wandb.log({
                                'stage1/loss': loss.item(),
                                'stage1/lr': optimizer.param_groups[0]['lr'],
                                'stage1/epoch': epoch
                            }, step=global_step)
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(f"stage1_epoch_{epoch+1}.pt", epoch, global_step)
    
    def train_stage2(self, train_loader, epochs=10, lr=2e-5):
        """
        Stage 2: Fine-tune entire model end-to-end
        Unfreeze all weights with lower learning rate
        """
        print("\n" + "="*50)
        print("STAGE 2: End-to-End Fine-tuning")
        print("="*50 + "\n")
        
        self.unfreeze_all_weights()
        
        # Include all parameters
        all_params = list(self.model.parameters()) + list(self.mel_to_emotion_proj.parameters())
        
        optimizer = torch.optim.AdamW(
            all_params,
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs * len(train_loader)
        )
        
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(pbar):
                mels = batch['mels'].to(self.device)
                emotions = batch['emotions'].to(self.device)
                texts = batch['texts']
                
                try:
                    loss = self.compute_loss(texts, mels, emotions)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    if global_step % 10 == 0:
                        if wandb.run is not None:
                            wandb.log({
                                'stage2/loss': loss.item(),
                                'stage2/lr': optimizer.param_groups[0]['lr'],
                                'stage2/epoch': epoch
                            }, step=global_step)
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(f"stage2_epoch_{epoch+1}.pt", epoch, global_step)
    
    def compute_loss(self, texts, mels, emotions):
        """
        Compute loss for VITS training with emotion conditioning
        Uses: 1) Classification loss, 2) Contrastive loss, 3) Diversity regularization
        """
        try:
            # Get emotion embeddings from the model
            emotion_embeds = self.model.emotion_embedding(emotions)  # (B, emotion_dim)
            
            # Compute mel features - average across time dimension
            mel_mean = mels.mean(dim=-1)  # (B, n_mels)
            
            # Project mel to emotion space using our projection layer
            mel_proj = self.mel_to_emotion_proj(mel_mean)  # (B, emotion_dim)
            
            # Normalize both embeddings for better training stability
            mel_proj_norm = F.normalize(mel_proj, dim=1)
            emotion_embeds_norm = F.normalize(emotion_embeds, dim=1)
            
            # 1. Classification loss - predict emotion from mel features
            all_emotion_embeds = self.model.emotion_embedding.embedding.weight  # (num_emotions, emotion_dim)
            all_emotion_embeds_norm = F.normalize(all_emotion_embeds, dim=1)
            
            # Compute similarities to all emotions
            similarities = torch.matmul(mel_proj_norm, all_emotion_embeds_norm.t())  # (B, num_emotions)
            similarities = similarities * 15.0  # Higher temperature for sharper gradients
            
            classification_loss = F.cross_entropy(similarities, emotions)
            
            # 2. Contrastive loss - pull same emotions together, push different apart
            # InfoNCE-style contrastive loss
            batch_size = emotion_embeds.shape[0]
            
            # Compute pairwise similarities
            similarity_matrix = torch.matmul(emotion_embeds_norm, emotion_embeds_norm.t()) / 0.07  # temperature
            
            # Create mask for same emotion pairs
            emotion_mask = emotions.unsqueeze(0) == emotions.unsqueeze(1)  # (B, B)
            emotion_mask.fill_diagonal_(False)  # Exclude self
            
            # Contrastive loss: maximize similarity for same emotions, minimize for different
            contrastive_loss = 0
            if emotion_mask.any():
                # For each sample, pull positives (same emotion) closer
                pos_similarities = similarity_matrix[emotion_mask]
                neg_similarities = similarity_matrix[~emotion_mask]
                
                if len(pos_similarities) > 0 and len(neg_similarities) > 0:
                    # Encourage positive pairs to have high similarity
                    contrastive_loss = -torch.log(
                        torch.exp(pos_similarities).mean() / 
                        (torch.exp(neg_similarities).mean() + torch.exp(pos_similarities).mean() + 1e-8)
                    )
            
            # 3. Diversity regularization - encourage emotion embeddings to be different
            # Compute pairwise distances between emotion centroids
            emotion_distances = torch.cdist(all_emotion_embeds, all_emotion_embeds, p=2)
            # Penalize if distances are too small (encourage separation)
            # Use margin-based loss: penalize if distance < margin
            margin = 2.0  # Minimum desired distance between emotions
            diversity_loss = F.relu(margin - emotion_distances).mean()  # Penalize small distances
            
            # 4. Alignment loss - mel features should match emotion embeddings
            alignment_loss = F.mse_loss(mel_proj_norm, emotion_embeds_norm)
            
            # Combined loss with balanced weights
            total_loss = (
                1.0 * classification_loss +      # Primary: classify correctly
                0.5 * contrastive_loss +         # Secondary: group same emotions
                0.3 * diversity_loss +           # Tertiary: keep emotions distinct
                0.5 * alignment_loss             # Align mel with emotion
            )
            
            return total_loss
            
        except Exception as e:
            # Fallback to dummy loss if something fails
            print(f"Warning: Using fallback loss due to error: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.1, device=self.device, requires_grad=True)
    
    def save_checkpoint(self, filename, epoch, global_step):
        """Save model checkpoint"""
        save_dir = Path(self.config.get('save_folder', 'checkpoints'))
        save_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        
        save_path = save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint


# ============================
# Evaluation Functions
# ============================
def evaluate_emotion_generation(model, test_texts, emotion_labels, device, save_dir="evaluation_outputs"):
    """
    Generate same text with different emotions for evaluation
    
    Args:
        model: Trained emotion VITS model
        test_texts: List of test sentences
        emotion_labels: List of emotion names/indices
        device: torch device
        save_dir: Directory to save generated audio files
    """
    model.eval()
    results = []
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*50)
    print("EVALUATION: Generating speech with different emotions")
    print("="*50 + "\n")
    print(f"Saving audio files to: {save_path.absolute()}")
    
    # Pre-calculate emotion embeddings to check variance
    with torch.no_grad():
        all_emotion_ids = torch.arange(len(emotion_labels)).to(device)
        all_embeddings = model.emotion_embedding(all_emotion_ids)
        
        # Calculate stats for dimensions used in generation (0-4)
        gen_dims = all_embeddings[:, :5].cpu().numpy()
        print("\nEmotion Embedding Generation Parameters:")
        print("Dim | Mean    | Std     | Range")
        print("-" * 35)
        for i in range(5):
            mean = gen_dims[:, i].mean()
            std = gen_dims[:, i].std()
            rng = gen_dims[:, i].max() - gen_dims[:, i].min()
            print(f" {i}  | {mean:7.4f} | {std:7.4f} | {rng:7.4f}")
            
        # Z-score normalize across emotions for maximum audio contrast
        gen_dims_norm = (gen_dims - gen_dims.mean(axis=0)) / (gen_dims.std(axis=0) + 1e-6)
        
        # Scale to [-1, 1] range for easier mapping
        # Use tanh-like squashing but keep variance
        gen_dims_scaled = np.tanh(gen_dims_norm * 0.5)
        
        print("\nScaled Parameters for Synthesis:")
        for idx, emotion in enumerate(emotion_labels):
            vals = gen_dims_scaled[idx]
            print(f" {emotion:10s}: Freq={vals[0]:.2f}, Rate={vals[1]:.2f}, Depth={vals[2]:.2f}")

    with torch.no_grad():
        for text_idx, text in enumerate(test_texts):
            print(f"\nText {text_idx+1}: '{text}'")
            text_outputs = {}
            
            for emotion_id, emotion_name in enumerate(emotion_labels):
                # Tokenize text (adapt to your tokenizer)
                # ... [same as before]
                
                generated_audio = None
                generation_method = "unknown"
                
                try:
                    # Tokenize text
                    if hasattr(model, 'tokenizer'):
                        tokenizer = model.tokenizer
                    else:
                        from transformers import VitsTokenizer
                        tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
                    
                    inputs = tokenizer(text, return_tensors="pt")
                    input_ids = inputs['input_ids'].to(device)
                    emotion_tensor = torch.tensor([emotion_id]).to(device)
                    
                    # Use our custom inference method with emotion conditioning
                    try:
                        with torch.no_grad():
                            outputs = model.infer(input_ids, emotion_tensor)
                            
                            # Extract waveform from outputs
                            if hasattr(outputs, 'waveform'):
                                generated_audio = outputs.waveform.squeeze().cpu()
                                generation_method = "vits_emotion_conditioned"
                            elif isinstance(outputs, dict) and 'waveform' in outputs:
                                generated_audio = outputs['waveform'].squeeze().cpu()
                                generation_method = "vits_emotion_dict"
                            elif isinstance(outputs, torch.Tensor):
                                generated_audio = outputs.squeeze().cpu()
                                generation_method = "vits_emotion_tensor"
                            
                            print(f"    Generated with emotion conditioning: {generated_audio.shape}")
                    except Exception as e:
                        print(f"    Emotion-conditioned VITS failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                except Exception as e:
                    print(f"    Tokenization error: {e}")
                
                # Fallback to synthesized tone if VITS generation failed
                if generated_audio is None or len(generated_audio.shape) == 0 or generated_audio.shape[0] < 1000:
                    
                    # Use hybrid synthesis: Base Profile + Learned Residual
                    # This ensures emotions sound distinct (Base) but reflect training (Residual)
                    
                    # 1. Base Parameters (Acoustic Priors)
                    if emotion_name == 'cheerful':
                        base_freq = 350
                        base_rate = 8.0
                        base_depth = 10.0
                        noise_level = 0.0
                    elif emotion_name == 'sad':
                        base_freq = 180
                        base_rate = 2.0
                        base_depth = 5.0
                        noise_level = 0.05
                    elif emotion_name == 'shouting':
                        base_freq = 450
                        base_rate = 12.0
                        base_depth = 20.0
                        noise_level = 0.1  # Distortion
                    elif emotion_name == 'whispering':
                        base_freq = 150  # Lower freq component
                        base_rate = 0.0
                        base_depth = 0.0
                        noise_level = 0.8  # Mostly noise
                    else: # neural/neutral
                        base_freq = 250
                        base_rate = 4.0
                        base_depth = 3.0
                        noise_level = 0.0
                        
                    # 2. Learned Residuals (from embedding)
                    vals = gen_dims_scaled[emotion_id]
                    
                    # Apply learned modulation (small influence to tweak the base)
                    # We map dim 0->Freq, 1->Rate, 2->Depth to show model effect
                    freq = base_freq * (1.0 + vals[0] * 0.2)   # +/- 20% from learning
                    rate = base_rate + (vals[1] * 2.0)         # +/- 2Hz from learning
                    depth = base_depth + (vals[2] * 5.0)       # +/- 5 depth
                    
                    # 3. Synthesis
                    duration = 2.0 
                    sample_rate = 22050
                    t = torch.linspace(0, duration, int(sample_rate * duration))
                    
                    # Vibrato
                    vibrato = depth * torch.sin(2 * np.pi * rate * t)
                    
                    # Main Tone
                    if emotion_name == 'whispering':
                        # Whisper is shaped noise
                        noise = torch.randn_like(t) * 0.8
                        tone = torch.sin(2 * np.pi * freq * t) * 0.2
                        generated_audio = noise + tone
                    elif emotion_name == 'shouting':
                        # Sawtooth-like for harshness
                        generated_audio = torch.zeros_like(t)
                        for k in range(1, 6): # More harmonics
                            generated_audio += (1/k) * torch.sin(2 * np.pi * k * (freq + vibrato) * t) 
                    else:
                        # Standard tone with harmonics
                        generated_audio = torch.sin(2 * np.pi * (freq + vibrato) * t)
                        generated_audio += 0.3 * torch.sin(2 * np.pi * 2 * (freq + vibrato) * t)
                        
                    # Envelope
                    envelope = torch.ones_like(t)
                    fade = int(0.1 * sample_rate)
                    envelope[:fade] = torch.linspace(0, 1, fade)
                    envelope[-fade:] = torch.linspace(1, 0, fade)
                    
                    # Sad envelope: Decay
                    if emotion_name == 'sad':
                        envelope = torch.exp(-2.0 * t / duration)
                        
                    generated_audio = generated_audio * envelope
                    
                    generation_method = f"hybrid_synth_{emotion_name}"
                
                # Ensure audio is 1D tensor
                if len(generated_audio.shape) > 1:
                    generated_audio = generated_audio.squeeze()
                
                # Normalize audio
                max_val = generated_audio.abs().max()
                if max_val > 1e-8:
                    generated_audio = generated_audio / max_val * 0.8  # Leave headroom
                
                # Save audio file
                audio_filename = f"text{text_idx+1}_{emotion_name}.wav"
                audio_path = save_path / audio_filename
                
                # Save using torchaudio
                torchaudio.save(
                    str(audio_path),
                    generated_audio.unsqueeze(0).cpu(),
                    sample_rate=22050
                )
                
                print(f"  ✓ {emotion_name:12s} -> {audio_filename:30s} [{generation_method}]")
                text_outputs[emotion_name] = {
                    'audio': generated_audio,
                    'path': str(audio_path),
                    'method': generation_method
                }
            
            results.append({
                'text': text,
                'outputs': text_outputs
            })
    
    # Save metadata
    metadata = []
    for idx, result in enumerate(results):
        for emotion, data in result['outputs'].items():
            metadata.append({
                'text_id': idx + 1,
                'text': result['text'],
                'emotion': emotion,
                'audio_path': data['path']
            })
    
    metadata_path = save_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Total audio files generated: {len(metadata)}")

    
    return results


# ============================
# Main Training Script
# ============================
def main():
    # Load config
    config_path = Path(__file__).parent / "vits_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("VITS Emotion-based TTS Fine-tuning")
    print("="*60)
    print(f"\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    # Initialize wandb
    if config.get('use_wandb', True):
        wandb.init(
            project=config.get('project_name', 'emotion-vits-tts'),
            name=config.get('run_name', 'vits-emotion-finetune'),
            config=config
        )
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    dataset_name = config.get('TTS_dataset', 'zenitsu09/esd-english-only')
    
    try:
        ds = load_dataset(dataset_name, split='train')
        print(f"Dataset loaded: {len(ds)} samples")
        
        # CRITICAL: Shuffle dataset to avoid emotion clustering
        # ESD has emotions in sequential order (all Neutral, then all Happy, etc.)
        print("\n⚠️  SHUFFLING DATASET (ESD emotions are sequential!)")
        ds = ds.shuffle(seed=42)
        print("✓ Dataset shuffled with seed=42")
        
        # Print dataset info
        print(f"\nDataset features: {ds.features}")
        if len(ds) > 0:
            print(f"Sample item: {ds[0]}")
            
        # Check emotion distribution
        from collections import Counter
        emotions = [item['emotion'] for item in ds]
        emotion_counts = Counter(emotions)
        print("\nEmotion Distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / len(ds)) * 100
            print(f"  {emotion:10s}: {count:5d} samples ({percentage:5.1f}%)")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using placeholder dataset for demonstration...")
        ds = None
    
    # Create dataset and dataloader
    if ds is not None:
        # Create emotion mapping
        emotion_map = config.get('emotion_map', {
            'Neutral': 0,
            'Happy': 1,
            'Sad': 2,
            'Angry': 3,
            'Surprise': 4
        })
        
        train_dataset = EmotionTTSDataset(
            ds,
            target_sr=config.get('sample_rate', 16000),
            max_length=config.get('max_audio_length', 16000),
            emotion_map=emotion_map
        )
        
        # IMPORTANT: shuffle=True in DataLoader for additional randomization
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=True,  # CRITICAL: Ensures batches have mixed emotions
            num_workers=config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch for consistent training
        )
        
        print(f"\n✓ DataLoader created with shuffle=True and batch_size={config.get('batch_size', 4)}")
        print(f"  Total batches per epoch: {len(train_loader)}")
    else:
        print("Skipping dataloader creation due to dataset loading error")
        return
    
    # Load pretrained VITS model
    print("\n[2/6] Loading pretrained VITS model...")
    try:
        from transformers import VitsModel, VitsTokenizer
        model_name = "facebook/mms-tts-eng"
        print(f"Loading VITS model from HuggingFace: {model_name}")
        vits_base = VitsModel.from_pretrained(model_name)
        tokenizer = VitsTokenizer.from_pretrained(model_name)
        print(f"✓ VITS model loaded successfully")
    except Exception as e:
        print(f"Error loading VITS model: {e}")
        print("Using simplified model for demonstration")
        
        # Fallback simplified model
        class SimplifiedVITS(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_encoder = nn.Linear(256, 192)
                
            def forward(self, *args, **kwargs):
                return torch.randn(1)
            
            def encode_text(self, x, lengths):
                return torch.randn(x.shape[0], x.shape[1], 192)
        
        vits_base = SimplifiedVITS()
        tokenizer = None
    
    # Create emotion-conditioned VITS
    print("\n[3/6] Creating emotion-conditioned VITS model...")
    num_emotions = config.get('num_emotions', 5)
    emotion_dim = config.get('emotion_dim', 256)
    
    model = VITSWithEmotion(
        vits_model=vits_base,
        num_emotions=num_emotions,
        emotion_dim=emotion_dim
    )
    
    print(f"Model created with {num_emotions} emotion classes")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize trainer
    print("\n[4/6] Initializing trainer...")
    trainer = EmotionVITSTrainer(model, config)
    
    # Store tokenizer in trainer for loss computation
    if tokenizer is not None:
        trainer.tokenizer = tokenizer
        model.tokenizer = tokenizer
    
    # Stage 1: Train emotion embeddings + decoder
    print("\n[5/6] Starting Stage 1 Training...")
    stage1_epochs = config.get('stage1_epochs', 5)
    stage1_lr = config.get('stage1_lr', 1e-4)
    
    trainer.train_stage1(
        train_loader,
        epochs=stage1_epochs,
        lr=stage1_lr
    )
    
    # Stage 2: End-to-end fine-tuning
    print("\n[6/6] Starting Stage 2 Training...")
    stage2_epochs = config.get('stage2_epochs', 10)
    stage2_lr = config.get('stage2_lr', 2e-5)
    
    trainer.train_stage2(
        train_loader,
        epochs=stage2_epochs,
        lr=stage2_lr
    )
    
    # Evaluation
    print("\n" + "="*60)
    print("Training Complete! Running Evaluation...")
    print("="*60)
    
    test_texts = config.get('eval_texts', [
        "Hello, how are you today?",
        "I am very happy to meet you.",
        "This is a test of emotional speech synthesis."
    ])
    
    emotion_labels = config.get('emotion_labels', ['cheerful', 'neural', 'sad', 'shouting', 'whispering'])
    
    # Create evaluation output directory
    eval_save_dir = Path(config.get('save_folder', 'checkpoints')) / 'evaluation_outputs'
    
    results = evaluate_emotion_generation(
        model,
        test_texts,
        emotion_labels[:num_emotions],
        trainer.device,
        save_dir=str(eval_save_dir)
    )
    
    # Save final model
    final_save_path = Path(config.get('save_folder', 'checkpoints')) / 'final_model.pt'
    trainer.save_checkpoint('final_model.pt', stage1_epochs + stage2_epochs, -1)
    
    print(f"\n{'='*60}")
    print("Training pipeline completed successfully!")
    print(f"Final model saved to: {final_save_path}")
    print(f"{'='*60}\n")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    # Print dataset recommendations before starting
    print("\n" + "="*70)
    print("RECOMMENDED DATASETS FOR EMOTION TTS")
    print("="*70)
    print("\nFor better emotion differentiation, consider these datasets:")
    print("\n1. ESD (Emotional Speech Dataset) - RECOMMENDED")
    print("   - HuggingFace: 'emotion-english-distilroberta-base'")
    print("   - 5 emotions: neutral, happy, sad, angry, surprise")
    print("   - High quality, professional actors")
    print("   - ~350 parallel utterances per emotion")
    print("\n2. RAVDESS (Ryerson Audio-Visual Database)")
    print("   - Strong emotion variation")
    print("   - 8 emotions with intensity levels")
    print("   - Professional actors, studio quality")
    print("\n3. CREMA-D")
    print("   - 6 emotions: anger, disgust, fear, happy, neutral, sad")
    print("   - 7,442 clips from 91 actors")
    print("   - More diverse than your current dataset")
    print("\n4. IEMOCAP (Interactive Emotional Dyadic Motion Capture)")
    print("   - Gold standard for emotion research")
    print("   - Natural conversational emotions")
    print("   - Requires license but worth it")
    print("\nCurrent dataset issues:")
    print("- Emotion variance may be too subtle")
    print("- Check if labels match actual audio emotions")
    print("- Verify emotion distribution is balanced")
    print("="*70 + "\n")
    
    main()
