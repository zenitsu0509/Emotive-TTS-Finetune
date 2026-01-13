"""
Inference script for Emotion-based VITS TTS
Generate speech with different emotions using the fine-tuned model
"""

import torch
import torchaudio
import yaml
from pathlib import Path
import argparse
from vits_emotion_finetune import VITSWithEmotion, EmotionEmbedding
from vits_model_utils import load_pretrained_vits, TextProcessor, AudioProcessor


class EmotionTTSInference:
    """Inference engine for emotion-based TTS"""
    
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.get('sample_rate', 22050)
        )
        
        # Emotion mapping
        self.emotion_labels = self.config.get('emotion_labels', [
            'neutral', 'happy', 'sad', 'angry', 'surprise'
        ])
        self.emotion_to_id = {label: i for i, label in enumerate(self.emotion_labels)}
        
        print(f"Model loaded on {self.device}")
        print(f"Available emotions: {', '.join(self.emotion_labels)}")
    
    def _load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate model architecture
        num_emotions = self.config.get('num_emotions', 5)
        emotion_dim = self.config.get('emotion_dim', 256)
        
        # Load base VITS (this is simplified - you need actual VITS model)
        # vits_base = load_pretrained_vits(self.config)
        
        # For demonstration, using placeholder
        import torch.nn as nn
        class DummyVITS(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Linear(10, 10)
            
            def infer(self, *args, **kwargs):
                # Return dummy audio (1 second at 22050 Hz)
                return torch.randn(1, 22050)
        
        vits_base = DummyVITS()
        
        model = VITSWithEmotion(
            vits_model=vits_base,
            num_emotions=num_emotions,
            emotion_dim=emotion_dim
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def synthesize(self, text, emotion='neutral', noise_scale=0.667, length_scale=1.0):
        """
        Synthesize speech with specified emotion
        
        Args:
            text: Input text to synthesize
            emotion: Emotion label or ID
            noise_scale: Variance for stochastic duration predictor
            length_scale: Duration scaling factor
        
        Returns:
            audio: Generated audio waveform (numpy array)
            sample_rate: Audio sample rate
        """
        # Convert emotion label to ID
        if isinstance(emotion, str):
            if emotion not in self.emotion_to_id:
                print(f"Warning: Unknown emotion '{emotion}', using 'neutral'")
                emotion = 'neutral'
            emotion_id = self.emotion_to_id[emotion]
        else:
            emotion_id = emotion
        
        emotion_id = torch.tensor([emotion_id]).to(self.device)
        
        # Process text
        text_tokens = self.text_processor.process_text(text)
        text_tokens = text_tokens.to(self.device)
        text_lengths = torch.tensor([text_tokens.shape[1]]).to(self.device)
        
        # Generate audio
        with torch.no_grad():
            audio = self.model.infer(
                text_tokens,
                text_lengths,
                emotion_id,
                noise_scale=noise_scale,
                length_scale=length_scale
            )
        
        # Convert to numpy
        audio = audio.squeeze().cpu().numpy()
        
        return audio, self.config.get('sample_rate', 22050)
    
    def synthesize_all_emotions(self, text, output_dir='outputs'):
        """
        Generate speech for all emotions
        
        Args:
            text: Input text
            output_dir: Directory to save outputs
        
        Returns:
            Dictionary mapping emotions to audio file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = {}
        
        print(f"\nGenerating speech for: '{text}'")
        print("=" * 60)
        
        for emotion in self.emotion_labels:
            print(f"Synthesizing with emotion: {emotion}...")
            
            audio, sr = self.synthesize(text, emotion)
            
            # Save audio
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            output_path = output_dir / f"{emotion}.wav"
            torchaudio.save(str(output_path), audio_tensor, sr)
            
            results[emotion] = output_path
            print(f"  ✓ Saved to {output_path}")
        
        print("=" * 60)
        print(f"All audio files saved to {output_dir}")
        
        return results
    
    def interactive_mode(self):
        """Interactive mode for testing"""
        print("\n" + "=" * 60)
        print("Interactive Emotion TTS")
        print("=" * 60)
        print("\nCommands:")
        print("  - Type text to synthesize")
        print("  - Prefix with emotion: 'happy: Hello there!'")
        print("  - Type 'emotions' to see available emotions")
        print("  - Type 'quit' to exit")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'emotions':
                    print(f"Available emotions: {', '.join(self.emotion_labels)}")
                    continue
                
                # Parse emotion and text
                if ':' in user_input:
                    emotion, text = user_input.split(':', 1)
                    emotion = emotion.strip()
                    text = text.strip()
                else:
                    emotion = 'neutral'
                    text = user_input
                
                # Synthesize
                print(f"Generating with emotion '{emotion}'...")
                audio, sr = self.synthesize(text, emotion)
                
                # Save
                output_path = Path('outputs') / 'interactive' / f"last_output.wav"
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                audio_tensor = torch.tensor(audio).unsqueeze(0)
                torchaudio.save(str(output_path), audio_tensor, sr)
                
                print(f"✓ Audio saved to {output_path}")
                print(f"  Duration: {len(audio) / sr:.2f}s\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description='Emotion-based VITS TTS Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='vits_config.yaml',
                        help='Path to config file')
    parser.add_argument('--text', type=str,
                        help='Text to synthesize')
    parser.add_argument('--emotion', type=str, default='neutral',
                        help='Emotion for synthesis')
    parser.add_argument('--all-emotions', action='store_true',
                        help='Generate with all emotions')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    print("Initializing Emotion TTS Inference...")
    inference = EmotionTTSInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Interactive mode
    if args.interactive:
        inference.interactive_mode()
        return
    
    # Batch mode
    if args.text:
        if args.all_emotions:
            # Generate with all emotions
            results = inference.synthesize_all_emotions(args.text, args.output_dir)
            print(f"\nGenerated {len(results)} audio files")
        else:
            # Generate with single emotion
            audio, sr = inference.synthesize(args.text, args.emotion)
            
            # Save
            output_path = Path(args.output_dir) / f"{args.emotion}.wav"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            torchaudio.save(str(output_path), audio_tensor, sr)
            
            print(f"Audio saved to {output_path}")
    else:
        print("Please provide --text or use --interactive mode")


if __name__ == "__main__":
    main()
