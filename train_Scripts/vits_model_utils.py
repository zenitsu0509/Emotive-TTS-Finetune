"""
VITS Model Integration for Emotion-based TTS
This file provides integration helpers for different VITS implementations
"""

import torch
import torch.nn as nn


# ============================
# Option 1: Using TTS library (Coqui/Mozilla TTS)
# ============================
def load_vits_from_tts_library(model_path, config_path):
    """
    Load VITS model from TTS library
    
    Installation:
        pip install TTS
    
    Usage:
        vits_model = load_vits_from_tts_library("path/to/model.pth", "path/to/config.json")
    """
    try:
        from TTS.tts.models.vits import Vits
        from TTS.tts.configs.vits_config import VitsConfig
        
        config = VitsConfig()
        config.load_json(config_path)
        
        model = Vits(config)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        return model, config
    except ImportError:
        print("TTS library not installed. Run: pip install TTS")
        return None, None


# ============================
# Option 2: Using Hugging Face Transformers
# ============================
def load_vits_from_huggingface(model_name="facebook/mms-tts-eng"):
    """
    Load VITS-based model from Hugging Face
    
    Installation:
        pip install transformers
    
    Usage:
        model, processor = load_vits_from_huggingface("facebook/mms-tts-eng")
    """
    try:
        from transformers import VitsModel, VitsTokenizer
        
        model = VitsModel.from_pretrained(model_name)
        tokenizer = VitsTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
    except ImportError:
        print("Transformers library not installed. Run: pip install transformers")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# ============================
# Option 3: Custom VITS Integration Wrapper
# ============================
class VITSModelWrapper(nn.Module):
    """
    Wrapper to standardize different VITS implementations
    Provides a common interface for emotion-based fine-tuning
    """
    
    def __init__(self, vits_model, model_type='tts_library'):
        """
        Args:
            vits_model: The loaded VITS model
            model_type: 'tts_library', 'huggingface', or 'custom'
        """
        super().__init__()
        self.vits = vits_model
        self.model_type = model_type
    
    def encode_text(self, text_inputs, text_lengths):
        """
        Encode text to hidden representations
        
        Args:
            text_inputs: Tokenized text (B, T)
            text_lengths: Text sequence lengths (B,)
        
        Returns:
            text_encoded: Encoded text features (B, T, D)
        """
        if self.model_type == 'tts_library':
            # TTS library uses text encoder
            if hasattr(self.vits, 'text_encoder'):
                return self.vits.text_encoder(text_inputs, text_lengths)
            else:
                raise NotImplementedError("Text encoder not found in model")
        
        elif self.model_type == 'huggingface':
            # Hugging Face VITS
            if hasattr(self.vits, 'text_encoder'):
                outputs = self.vits.text_encoder(text_inputs)
                return outputs.last_hidden_state
            else:
                raise NotImplementedError("Text encoder not found in model")
        
        elif self.model_type == 'custom':
            # Implement custom text encoding
            return self.vits.encode_text(text_inputs, text_lengths)
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def forward(self, text_inputs, text_lengths, audio_targets, audio_lengths, text_encoded=None):
        """
        Forward pass for training
        
        Args:
            text_inputs: Tokenized text
            text_lengths: Text lengths
            audio_targets: Target mel spectrograms
            audio_lengths: Audio lengths
            text_encoded: Pre-encoded text (if using emotion conditioning)
        
        Returns:
            Dictionary with loss components
        """
        if self.model_type == 'tts_library':
            # TTS library forward pass
            outputs = self.vits(
                text_inputs,
                text_lengths,
                audio_targets,
                audio_lengths
            )
            return outputs
        
        elif self.model_type == 'huggingface':
            # Hugging Face forward pass
            outputs = self.vits(
                input_ids=text_inputs,
                attention_mask=text_lengths,
                labels=audio_targets
            )
            return {'loss': outputs.loss}
        
        elif self.model_type == 'custom':
            return self.vits(text_inputs, text_lengths, audio_targets, audio_lengths)
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def infer(self, text_inputs, text_lengths, text_encoded=None, noise_scale=0.667, length_scale=1.0):
        """
        Inference/generation
        
        Args:
            text_inputs: Tokenized text
            text_lengths: Text lengths
            text_encoded: Pre-encoded text with emotion conditioning
            noise_scale: Variance for stochastic duration predictor
            length_scale: Duration scaling factor
        
        Returns:
            Generated audio waveform
        """
        if self.model_type == 'tts_library':
            if hasattr(self.vits, 'inference'):
                return self.vits.inference(
                    text_inputs,
                    aux_input={'noise_scale': noise_scale, 'length_scale': length_scale}
                )
            else:
                raise NotImplementedError("Inference method not found")
        
        elif self.model_type == 'huggingface':
            with torch.no_grad():
                outputs = self.vits.generate(
                    input_ids=text_inputs,
                    attention_mask=text_lengths
                )
                return outputs
        
        elif self.model_type == 'custom':
            return self.vits.infer(text_inputs, text_lengths)
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


# ============================
# Text Processing Utilities
# ============================
class TextProcessor:
    """Text processing for VITS models"""
    
    def __init__(self, tokenizer=None, use_phonemes=False):
        """
        Args:
            tokenizer: Tokenizer instance (optional)
            use_phonemes: Whether to use phoneme-based tokenization
        """
        self.tokenizer = tokenizer
        self.use_phonemes = use_phonemes
        
        if use_phonemes:
            self._init_phonemizer()
    
    def _init_phonemizer(self):
        """Initialize phonemizer for better pronunciation"""
        try:
            from phonemizer import phonemize
            self.phonemize = phonemize
        except ImportError:
            print("Phonemizer not installed. Run: pip install phonemizer")
            self.phonemize = None
    
    def process_text(self, text):
        """
        Process text to model input format
        
        Args:
            text: Input text string
        
        Returns:
            Tokenized text tensor
        """
        if self.use_phonemes and self.phonemize is not None:
            # Convert to phonemes
            text = self.phonemize(text, language='en-us', backend='espeak')
        
        if self.tokenizer is not None:
            # Use provided tokenizer
            tokens = self.tokenizer(text, return_tensors='pt')
            return tokens['input_ids']
        else:
            # Simple character-level tokenization
            char_to_id = self._get_char_to_id()
            token_ids = [char_to_id.get(c, 0) for c in text.lower()]
            return torch.tensor([token_ids])
    
    def _get_char_to_id(self):
        """Simple character to ID mapping"""
        chars = " abcdefghijklmnopqrstuvwxyz.,!?'-"
        return {c: i for i, c in enumerate(chars)}


# ============================
# Audio Processing Utilities
# ============================
class AudioProcessor:
    """Audio processing for VITS training"""
    
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_transform = self._create_mel_transform()
    
    def _create_mel_transform(self):
        """Create mel spectrogram transform"""
        import torchaudio
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=0,
            f_max=8000
        )
    
    def audio_to_mel(self, waveform):
        """
        Convert audio waveform to mel spectrogram
        
        Args:
            waveform: Audio tensor (B, T) or (T,)
        
        Returns:
            mel: Mel spectrogram (B, n_mels, T') or (n_mels, T')
        """
        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))  # Log compression
        return mel
    
    def resample_audio(self, waveform, orig_sr):
        """
        Resample audio to target sample rate
        
        Args:
            waveform: Audio tensor
            orig_sr: Original sample rate
        
        Returns:
            Resampled audio
        """
        if orig_sr == self.sample_rate:
            return waveform
        
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
        return resampler(waveform)


# ============================
# Model Loading Helper
# ============================
def load_pretrained_vits(config):
    """
    Load pretrained VITS model based on configuration
    
    Args:
        config: Dictionary with model configuration
    
    Returns:
        vits_model: Loaded VITS model
        text_processor: Text processor
    """
    model_path = config.get('vits_pretrained_path')
    model_type = config.get('vits_model_type', 'tts_library')
    
    print(f"Loading VITS model from: {model_path}")
    print(f"Model type: {model_type}")
    
    if model_type == 'tts_library':
        config_path = config.get('vits_config_path')
        vits_model, vits_config = load_vits_from_tts_library(model_path, config_path)
        
    elif model_type == 'huggingface':
        model_name = config.get('hf_model_name', 'facebook/mms-tts-eng')
        vits_model, tokenizer = load_vits_from_huggingface(model_name)
        text_processor = TextProcessor(tokenizer=tokenizer)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Wrap model
    wrapped_model = VITSModelWrapper(vits_model, model_type=model_type)
    
    # Create text processor
    if model_type != 'huggingface':
        use_phonemes = config.get('use_phonemes', False)
        text_processor = TextProcessor(use_phonemes=use_phonemes)
    
    return wrapped_model, text_processor


# ============================
# Testing Functions
# ============================
def test_model_loading():
    """Test different model loading methods"""
    print("Testing VITS model loading...")
    
    # Test 1: TTS library
    print("\n1. Testing TTS library loading...")
    try:
        model, config = load_vits_from_tts_library(
            "path/to/model.pth",
            "path/to/config.json"
        )
        if model is not None:
            print("✓ TTS library loading successful")
    except Exception as e:
        print(f"✗ TTS library loading failed: {e}")
    
    # Test 2: Hugging Face
    print("\n2. Testing Hugging Face loading...")
    try:
        model, tokenizer = load_vits_from_huggingface()
        if model is not None:
            print("✓ Hugging Face loading successful")
            print(f"   Model: {model.__class__.__name__}")
    except Exception as e:
        print(f"✗ Hugging Face loading failed: {e}")
    
    print("\nModel loading tests completed!")


if __name__ == "__main__":
    test_model_loading()
