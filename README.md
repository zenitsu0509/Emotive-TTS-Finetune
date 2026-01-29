# VITS Emotion-based TTS Fine-tuning

Complete implementation for fine-tuning VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) models with emotion conditioning.

## 🎯 Overview

This project implements a two-stage training approach for adding emotion control to pretrained VITS TTS models:

1. **Stage 1**: Train emotion embeddings and decoder (frozen encoder)
2. **Stage 2**: End-to-end fine-tuning with lower learning rate

## 🌟 Features

- ✨ **Emotion Embedding Layer**: Inject emotion information into VITS architecture
- 🎭 **5 Emotion Classes**: Neutral, Happy, Sad, Angry, Surprise
- 🔄 **Two-Stage Training**: Efficient progressive fine-tuning
- 📊 **WandB Integration**: Track training metrics in real-time
- 🎵 **22.05kHz Audio**: High-quality speech synthesis
- 🚀 **Flexible Model Loading**: Support for multiple VITS implementations

## 📋 Dataset

Using the [PromptTTS Emotion-Tagged Dataset](https://huggingface.co/datasets/WhissleAI/emotion-tagged-small-v1):

- **Audio**: 22.05 kHz, 0.86-1.3s duration
- **Text**: Transcriptions
- **Labels**: 5 emotion classes
- **Format**: Hugging Face datasets

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/zenitsu0509/Emotive-TTS-Finetune.git
cd Emotive-TTS-Finetune

# Install dependencies
pip install -r requirements_vits.txt

# Login to Hugging Face
huggingface-cli login

# (Optional) Login to WandB
wandb login
```

### 2. Configure Training

Edit `train_Scripts/vits_config.yaml`:

```yaml
# Update these paths
vits_pretrained_path: "path/to/pretrained/vits/model"
TTS_dataset: "WhissleAI/emotion-tagged-small-v1"

# Training settings
stage1_epochs: 5      # Emotion embedding training
stage2_epochs: 10     # Full fine-tuning
batch_size: 4
```

### 3. Train the Model

```bash
cd train_Scripts

# Run training
python vits_emotion_finetune.py
```

### 4. Inference

```bash
# Generate with single emotion
python inference.py \
    --checkpoint checkpoints/vits_emotion/final_model.pt \
    --config vits_config.yaml \
    --text "Hello, how are you today?" \
    --emotion happy

# Generate with all emotions
python inference.py \
    --checkpoint checkpoints/vits_emotion/final_model.pt \
    --config vits_config.yaml \
    --text "Hello, how are you today!" \
    --all-emotions

# Interactive mode
python inference.py \
    --checkpoint checkpoints/vits_emotion/final_model.pt \
    --config vits_config.yaml \
    --interactive
```

## 📁 Project Structure

```
Emotive-TTS-Finetune/
├── train_Scripts/
│   ├── vits_emotion_finetune.py   # Main training script
│   ├── vits_model_utils.py        # Model loading utilities
│   ├── inference.py               # Inference script
│   ├── vits_config.yaml           # Configuration file
│   └── config.yaml                # Original config (for reference)
├── requirements_vits.txt          # Dependencies
└── README_VITS.md                 # This file
```

## 🏗️ Architecture

### Emotion Embedding Module

```python
EmotionEmbedding(num_emotions=5, emotion_dim=256)
├── Embedding(5, 256)
└── Linear(256, 256) + ReLU
```

### VITS with Emotion Conditioning

```
Text → VITS Text Encoder → (B, T, 192)
                              ↓
Emotion → Emotion Embedding → (B, 256) → Expand → (B, T, 256)
                                                      ↓
                                            Concatenate
                                                      ↓
                                            Fusion Layer
                                                      ↓
                                            VITS Decoder → Audio
```

## 🎓 Training Details

### Stage 1: Emotion Embeddings + Decoder

- **Duration**: 5 epochs
- **Learning Rate**: 1e-4
- **Frozen**: Text encoder, flow modules
- **Trainable**: Emotion embedding, fusion layer, decoder

### Stage 2: End-to-End Fine-tuning

- **Duration**: 10 epochs
- **Learning Rate**: 2e-5 (lower)
- **Trainable**: All parameters

### Optimizer

- **Type**: AdamW
- **Betas**: (0.9, 0.98)
- **Weight Decay**: 0.01
- **Gradient Clipping**: 5.0

## 🔧 Model Loading Options

The project supports multiple VITS implementations:

### Option 1: TTS Library (Coqui/Mozilla)

```python
from vits_model_utils import load_vits_from_tts_library

vits_model, config = load_vits_from_tts_library(
    "path/to/model.pth",
    "path/to/config.json"
)
```

### Option 2: Hugging Face

```python
from vits_model_utils import load_vits_from_huggingface

model, tokenizer = load_vits_from_huggingface("facebook/mms-tts-eng")
```

### Option 3: Custom VITS

Implement your own VITS model and use the `VITSModelWrapper` class.

## 📊 Monitoring Training

### WandB Dashboard

Training metrics logged:
- Loss curves (Stage 1 & 2)
- Learning rate schedule
- Gradient norms
- Audio samples (if configured)

### Console Output

```
[1/6] Loading dataset...
Dataset loaded: 10000 samples

[2/6] Loading pretrained VITS model...
Model created with 5 emotion classes
Total parameters: 28,345,120

==================================================
STAGE 1: Training Emotion Embeddings + Decoder
==================================================

Freezing VITS encoder weights...
Trainable parameters: 5,234,688

Epoch 1/5: 100%|████████| 2500/2500 [12:34<00:00, 3.31it/s, loss=0.4523]
Epoch 1 - Average Loss: 0.4782
```

## 🎤 Evaluation

Generate the same text with different emotions:

```python
from inference import EmotionTTSInference

inference = EmotionTTSInference(
    checkpoint_path="checkpoints/vits_emotion/final_model.pt",
    config_path="vits_config.yaml"
)

# Compare emotions
for emotion in ['neutral', 'happy', 'sad', 'angry', 'surprise']:
    audio, sr = inference.synthesize(
        "I am very excited about this!",
        emotion=emotion
    )
    # Listen to the differences!
```

## 🔍 Advanced Configuration

### Custom Emotion Labels

Edit `vits_config.yaml`:

```yaml
num_emotions: 7
emotion_labels:
  - "neutral"
  - "happy"
  - "sad"
  - "angry"
  - "surprise"
  - "fear"
  - "disgust"
```

### Audio Processing

```yaml
sample_rate: 22050
n_mels: 80
n_fft: 1024
hop_length: 256
```

### Multi-GPU Training

```yaml
distributed: true
# Run with: torchrun --nproc_per_node=4 vits_emotion_finetune.py
```

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
stage1_batch_size: 2
stage2_batch_size: 2
```

### Audio Quality Issues

Adjust inference parameters:
```python
audio, sr = inference.synthesize(
    text,
    emotion='happy',
    noise_scale=0.667,    # Lower = more deterministic
    length_scale=1.0      # Adjust speech duration
)
```

### Model Loading Errors

Make sure you have the correct VITS model format. Check `vits_model_utils.py` for supported formats.

## 📚 References

- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [PromptTTS Dataset](https://huggingface.co/datasets/WhissleAI/emotion-tagged-small-v1)
- [TTS Library](https://github.com/coqui-ai/TTS)
- [Hugging Face VITS](https://huggingface.co/docs/transformers/model_doc/vits)

## 📝 Citation

```bibtex
@article{kim2021conditional,
  title={Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 💬 Support

For questions or issues:
- Open a GitHub issue
- Contact: [your-email@example.com]

---

**Happy Training! 🎉**
