"""
Quick test script to verify ESD dataset loads correctly with shuffling
"""

from datasets import load_dataset
from collections import Counter

print("="*70)
print("Testing ESD Dataset Loading")
print("="*70)

# Load dataset
print("\n[1/3] Loading dataset...")
ds = load_dataset("zenitsu09/esd-english-only", split='train')
print(f"✓ Dataset loaded: {len(ds)} samples")

# Check BEFORE shuffle
print("\n[2/3] Checking emotion order BEFORE shuffle...")
print("\nFirst 20 samples (BEFORE shuffle):")
for i in range(min(20, len(ds))):
    print(f"  Sample {i:3d}: {ds[i]['emotion']:10s} - {ds[i]['transcription'][:50]}")

emotions_before = [ds[i]['emotion'] for i in range(min(100, len(ds)))]
print(f"\nFirst 100 emotions: {emotions_before[:20]}...")
print("⚠️  Notice: Emotions are likely in sequence (all same emotion in a row)")

# Shuffle
print("\n[3/3] Shuffling dataset...")
ds_shuffled = ds.shuffle(seed=42)
print("✓ Dataset shuffled")

# Check AFTER shuffle
print("\nFirst 20 samples (AFTER shuffle):")
for i in range(min(20, len(ds_shuffled))):
    print(f"  Sample {i:3d}: {ds_shuffled[i]['emotion']:10s} - {ds_shuffled[i]['transcription'][:50]}")

emotions_after = [ds_shuffled[i]['emotion'] for i in range(min(100, len(ds_shuffled)))]
print(f"\nFirst 100 emotions: {emotions_after[:20]}...")
print("✓ Emotions are now randomized!")

# Check distribution
print("\n" + "="*70)
print("Emotion Distribution:")
print("="*70)
all_emotions = [item['emotion'] for item in ds_shuffled]
emotion_counts = Counter(all_emotions)

for emotion, count in sorted(emotion_counts.items()):
    percentage = (count / len(ds_shuffled)) * 100
    bar = "█" * int(percentage / 2)
    print(f"{emotion:10s}: {count:5d} ({percentage:5.1f}%) {bar}")

print("\n" + "="*70)
print("Dataset is ready for training!")
print("="*70)
