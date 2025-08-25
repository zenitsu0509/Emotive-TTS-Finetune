# Emotive TTS Fine-tuning

A comprehensive toolkit for fine-tuning Text-to-Speech (TTS) models with emotional expression capabilities. This project provides multiple training approaches including full fine-tuning and efficient LoRA (Low-Rank Adaptation) methods to enhance TTS models with emotional understanding.

## Features

- 🎭 **Emotion-aware TTS training** using emotion-tagged datasets
- 🚀 **Multiple training methods**:
  - Full model fine-tuning
  - LoRA (Low-Rank Adaptation) for efficient training
- ⚡ **Optimized training** with Flash Attention 2 and BF16 precision
- 📊 **Experiment tracking** with Weights & Biases (WandB) integration
- ⚙️ **Configurable parameters** via YAML configuration
- 🔄 **Model merging and saving** for LoRA adapters

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account and token

### Dependencies

```bash
pip install torch transformers datasets peft wandb pyyaml huggingface_hub
```

## Quick Start

### 1. Setup

1. Clone the repository:
```bash
git clone https://github.com/zenitsu0509/Emotive-TTS-Finetune.git
cd Emotive-TTS-Finetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # if available, or install manually as shown above
```

3. Login to Hugging Face:
```bash
huggingface-cli login
```

4. Setup WandB (optional but recommended):
```bash
wandb login
```

### 2. Configuration

Edit the `train_Scripts/config.yaml` file to customize your training:

```yaml
# Dataset and Model
TTS_dataset: <WhissleAI/emotion-tagged-small-v1>
model_name: "canopylabs/orpheus-tts-0.1-pretrained"

# Training Parameters
epochs: 1
batch_size: 1
learning_rate: 5.0e-5
save_steps: 5000

# Paths and Naming
save_folder: "checkpoints"
project_name: "tuning-orpheus"
run_name: "experiment-1"
```

### 3. Training Options

#### Option A: Full Fine-tuning

For complete model fine-tuning with maximum adaptation:

```bash
cd train_Scripts
python train.py
```

#### Option B: LoRA Fine-tuning (Recommended)

For efficient training with reduced memory requirements:

```bash
cd train_Scripts
python lora_train.py
```

#### Option C: Notebook-style Training

For interactive training (requires token input):

```bash
python finetune_tts.py
```

## Training Methods Explained

### Full Fine-tuning (`train.py`)
- Updates all model parameters
- Requires more GPU memory and time
- Best for maximum model adaptation
- Suitable when you have sufficient computational resources

### LoRA Fine-tuning (`lora_train.py`)
- Only trains small adapter layers (rank=32, alpha=64)
- Significantly reduced memory requirements
- Faster training and inference
- Maintains base model performance while adding emotional capabilities
- Automatically merges and saves the final model

### Configuration Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `TTS_dataset` | Hugging Face dataset name | `WhissleAI/emotion-tagged-small-v1` |
| `model_name` | Base TTS model to fine-tune | `canopylabs/orpheus-tts-0.1-pretrained` |
| `epochs` | Number of training epochs | 1 |
| `batch_size` | Training batch size | 1 |
| `learning_rate` | Learning rate for optimization | 5.0e-5 |
| `save_steps` | Steps between model checkpoints | 5000 |
| `save_folder` | Directory for saving checkpoints | `checkpoints` |
| `project_name` | WandB project name | `tuning-orpheus` |
| `run_name` | WandB run identifier | `5e5-0` |

## Model and Dataset Details

### Base Model
- **Model**: `canopylabs/orpheus-tts-0.1-pretrained`
- **Type**: Causal Language Model adapted for TTS
- **Features**: Flash Attention 2 support, BF16 precision

### Training Dataset
- **Dataset**: `WhissleAI/emotion-tagged-small-v1`
- **Content**: Emotion-tagged text data for TTS training
- **Split**: Uses training split by default

## Output

### Full Fine-tuning Output
- Model checkpoints saved in `./checkpoints/`
- Complete fine-tuned model ready for inference

### LoRA Fine-tuning Output
- LoRA adapter checkpoints in `./checkpoints/`
- Merged final model in `./checkpoints/merged/`
- Both model weights and tokenizer saved

## Monitoring Training

All training runs automatically log to Weights & Biases with:
- Training loss progression
- Learning rate schedules
- System metrics (GPU usage, memory, etc.)
- Configurable run names for experiment organization

Access your experiments at: `https://wandb.ai/your-username/tuning-orpheus`

## Memory Requirements

### Full Fine-tuning
- Requires ~8-16GB GPU memory (depending on model size)
- Suitable for high-end GPUs (RTX 3080+, A100, etc.)

### LoRA Fine-tuning
- Requires ~4-8GB GPU memory
- Suitable for mid-range GPUs (RTX 3060+)
- Recommended for most users

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config.yaml
   - Use LoRA training instead of full fine-tuning
   - Enable gradient checkpointing

2. **Tokenizer Issues**
   - Ensure Hugging Face token has proper permissions
   - Verify model access rights

3. **WandB Integration**
   - Login with `wandb login` before training
   - Set `report_to: null` in training args to disable WandB

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Orpheus TTS](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained) for the base model
- [WhissleAI](https://huggingface.co/WhissleAI) for the emotion-tagged dataset
- Hugging Face team for transformers and PEFT libraries