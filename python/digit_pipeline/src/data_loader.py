"""
Raw Waveform Data Loader for Option A (12-Class German Digits)

Loads audio files as raw waveforms (16000 samples @ 16kHz) without external MFCC.
MFCC preprocessing is performed inside the model graph.

Dataset Structure:
- Flat file structure: data/hd_audio/audio_deutsch/*.flac
- Filename format: lang-german_speaker-XX_trial-YY_digit-Z.flac
- Labels extracted from filename (digit-Z)
- Speaker ID extracted for speaker-based train/val/test split

Classes (12):
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, _silence_, _unknown_
    Index mapping: digit → index, _silence_ → 10, _unknown_ → 11
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import librosa
from collections import defaultdict
import random

from digit_pipeline.config.config import TrainingConfig, AugmentationConfig


class AudioFileInfo:
    """Metadata for a single audio file"""
    def __init__(self, file_path: Path, speaker_id: str, label: int):
        self.file_path = file_path
        self.speaker_id = speaker_id
        self.label = label
    
    def __repr__(self):
        return f"AudioFileInfo(speaker={self.speaker_id}, label={self.label}, file={self.file_path.name})"


class GermanDigitRawWaveformLoader:
    """
    Data loader for German digit audio files (raw waveforms).
    
    Features:
    - Parses flat directory structure with filename-based labels
    - Extracts speaker ID from filename for speaker-based splitting
    - Loads audio as raw waveform (NO external MFCC)
    - Resamples to 16kHz, converts to mono, normalizes to [-1,1]
    - Pads/truncates to exactly 16000 samples (1 second)
    - Supports 12 classes: digits 0-9, _silence_, _unknown_
    - Augmentation: Time-shift, background-noise-mix, gain (training only)
    """
    
    # Use centralized config
    SAMPLE_RATE = TrainingConfig.SAMPLE_RATE
    TARGET_LENGTH = TrainingConfig.NUM_SAMPLES
    TIME_SHIFT_RANGE = AugmentationConfig.TIME_SHIFT_RANGE
    NOISE_MIX_PROB = AugmentationConfig.NOISE_MIX_PROBABILITY
    NOISE_VOLUME_RANGE = AugmentationConfig.NOISE_VOLUME_RANGE
    GAIN_RANGE = AugmentationConfig.GAIN_RANGE
    
    # Filename patterns
    # Digits: lang-german_speaker-XX_trial-YY_digit-Z.flac
    # Silence: lang-german_speaker-99_trial-XXX_digit-silence.flac
    # Unknown: lang-german_speaker-98_trial-XXX_digit-unknown.flac
    DIGIT_PATTERN = re.compile(
        r'lang-german_speaker-(\d+)_trial-\d+_digit-(\d+)\.flac'
    )
    SILENCE_PATTERN = re.compile(
        r'lang-german_speaker-(\d+)_trial-\d+_digit-silence\.flac'
    )
    UNKNOWN_PATTERN = re.compile(
        r'lang-german_speaker-(\d+)_trial-\d+_digit-unknown\.flac'
    )
    
    def __init__(self, audio_dir: str, labels_file: Optional[str] = None, noise_dir: Optional[str] = None, 
                 speech_commands_dir: Optional[str] = None, unknown_samples_per_class: int = 50,
                 mydigits_dir: Optional[str] = None,
                 mydigits_android_dir: Optional[str] = None,
                 mydigits_pc_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            audio_dir: Path to directory containing FLAC files
            labels_file: Optional path to labels.txt (0-9, _silence_, _unknown_)
            noise_dir: Optional path to noise directory for augmentation
            speech_commands_dir: Optional path to Speech Commands dataset for _unknown_ training
            unknown_samples_per_class: Number of samples per Speech Commands class for _unknown_
            mydigits_dir: DEPRECATED - use mydigits_android_dir instead
            mydigits_android_dir: Path to MyDigits Android recordings
            mydigits_pc_dir: Path to MyDigits PC recordings
        """
        self.audio_dir = Path(audio_dir)
        self.labels_file = Path(labels_file) if labels_file else None
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.speech_commands_dir = Path(speech_commands_dir) if speech_commands_dir else None
        self.unknown_samples_per_class = unknown_samples_per_class
        
        # MyDigits directories
        # Support both old single-directory and new split directories
        if mydigits_android_dir:
            self.mydigits_android_dir = Path(mydigits_android_dir)
        elif mydigits_dir:
            # Backward compatibility: treat old mydigits_dir as android dir
            self.mydigits_android_dir = Path(mydigits_dir)
        else:
            self.mydigits_android_dir = None
        self.mydigits_pc_dir = Path(mydigits_pc_dir) if mydigits_pc_dir else None
        
        # Load label mapping
        self.label_names = self._load_labels()
        self.num_classes = len(self.label_names)
        
        # Scan dataset
        self.file_infos: List[AudioFileInfo] = []
        self.speaker_ids: set = set()
        self._scan_dataset()
        
        # Add Speech Commands as _unknown_ (label 11) - Phase 4C
        if self.speech_commands_dir and self.speech_commands_dir.exists():
            self._add_speech_commands_unknown()
        
        # Add MyDigits manual recordings - Android + PC
        if self.mydigits_android_dir and self.mydigits_android_dir.exists():
            self._add_mydigits_source(self.mydigits_android_dir, "android")
        if self.mydigits_pc_dir and self.mydigits_pc_dir.exists():
            self._add_mydigits_source(self.mydigits_pc_dir, "pc")
        
        # Load noise samples for augmentation
        self.noise_samples: List[np.ndarray] = []
        if self.noise_dir and self.noise_dir.exists():
            self._load_noise_samples()
        
        print(f"[OK] Loaded {len(self.file_infos)} files from {len(self.speaker_ids)} speakers")
        print(f"[OK] Classes: {self.num_classes} ({', '.join(self.label_names)})")
    
    def _load_labels(self) -> List[str]:
        """Load label names from labels.txt or use default"""
        if self.labels_file and self.labels_file.exists():
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
            print(f"[OK] Loaded {len(labels)} labels from {self.labels_file}")
            return labels
        else:
            # Default: 12 classes (0-9, _silence_, _unknown_)
            default_labels = [str(i) for i in range(10)] + ['_silence_', '_unknown_']
            print(f"[WARN] labels.txt not found, using default: {default_labels}")
            return default_labels
    
    def _parse_filename(self, filename: str) -> Optional[Tuple[str, int]]:
        """
        Parse filename to extract speaker_id and label.
        
        Supports:
        - Digits 0-9: lang-german_speaker-XX_trial-YY_digit-Z.flac
        - Silence: lang-german_speaker-99_trial-XXX_digit-silence.flac (label 10)
        - Unknown: lang-german_speaker-98_trial-XXX_digit-unknown.flac (label 11)
        
        Args:
            filename: e.g., 'lang-german_speaker-05_trial-22_digit-9.flac'
        
        Returns:
            (speaker_id, label) or None if pattern doesn't match
        """
        # Try digit pattern first
        match = self.DIGIT_PATTERN.match(filename)
        if match:
            speaker_id = match.group(1)
            digit = int(match.group(2))
            return (speaker_id, digit)
        
        # Try silence pattern
        match = self.SILENCE_PATTERN.match(filename)
        if match:
            speaker_id = match.group(1)
            return (speaker_id, 10)  # _silence_ is class 10
        
        # Try unknown pattern
        match = self.UNKNOWN_PATTERN.match(filename)
        if match:
            speaker_id = match.group(1)
            return (speaker_id, 11)  # _unknown_ is class 11
        
        return None
    
    def _scan_dataset(self):
        """Scan audio directory and build file list with metadata"""
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")
        
        flac_files = list(self.audio_dir.glob("*.flac"))
        print(f"Found {len(flac_files)} FLAC files in {self.audio_dir}")
        
        for flac_file in flac_files:
            parsed = self._parse_filename(flac_file.name)
            if parsed:
                speaker_id, label = parsed
                
                # Include all labels 0-11 (digits 0-9, _silence_, _unknown_)
                if 0 <= label <= 11:
                    file_info = AudioFileInfo(flac_file, speaker_id, label)
                    self.file_infos.append(file_info)
                    self.speaker_ids.add(speaker_id)
            else:
                print(f"[WARN] Skipping file with unexpected format: {flac_file.name}")
        
        if not self.file_infos:
            raise ValueError(f"No valid audio files found in {self.audio_dir}")
    
    def _add_speech_commands_unknown(self):
        """
        Add Speech Commands non-digit classes as _unknown_ (label 11) - Phase 4C.
        
        Loads samples from Speech Commands dataset (yes, no, up, down, left, right, etc.)
        EXCLUDES digit words (zero, one, two, three, four, five, six, seven, eight, nine).
        All loaded samples get label 11 (_unknown_) and speaker_id 'sc_unknown'.
        """
        # Digit words to EXCLUDE (these are German digit training data)
        digit_words = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}
        
        # Scan all subdirectories
        all_classes = [d for d in self.speech_commands_dir.iterdir() 
                      if d.is_dir() and d.name not in digit_words and not d.name.startswith('_')]
        
        print(f"\n[OK] Phase 4C: Adding Speech Commands _unknown_ from {len(all_classes)} non-digit classes")
        print(f"  Sampling {self.unknown_samples_per_class} clips per class")
        
        total_added = 0
        rng = random.Random(42)  # Fixed seed for reproducibility
        
        for class_dir in sorted(all_classes):
            # Get all wav files in this class
            wav_files = list(class_dir.glob("*.wav"))
            
            if not wav_files:
                continue
            
            # Sample N files from this class
            n_samples = min(self.unknown_samples_per_class, len(wav_files))
            sampled_files = rng.sample(wav_files, n_samples)
            
            # Add to file_infos with label 11 (_unknown_)
            for wav_file in sampled_files:
                file_info = AudioFileInfo(
                    file_path=wav_file,
                    speaker_id='sc_unknown',  # Virtual speaker for Speech Commands unknowns
                    label=11  # _unknown_
                )
                self.file_infos.append(file_info)
                total_added += 1
        
        # Add virtual speaker
        self.speaker_ids.add('sc_unknown')
        
        print(f"  [OK] Added {total_added} Speech Commands samples as _unknown_ (label 11)")
        print(f"  Classes used: {', '.join(sorted([d.name for d in all_classes]))}")
    
    def _add_mydigits_source(self, source_dir: Path, source_name: str):
        """
        Add MyDigits manual recordings from a specific source (Android/PC).
        
        Directory structure: source_dir/0-9/*.wav
        Each source gets its own virtual speaker_id for proper train/val/test splitting.
        
        Args:
            source_dir: Path to the mydigits source directory
            source_name: Name of the source ("android" or "pc")
        """
        speaker_id = f'mydigits_{source_name}'  # e.g., 'mydigits_android', 'mydigits_pc'
        
        print(f"\n[OK] Adding MyDigits {source_name.upper()} recordings from {source_dir}")
        
        total_added = 0
        per_digit_count = {d: 0 for d in range(10)}
        
        # Load digit folders 0-9
        for digit in range(10):
            digit_dir = source_dir / str(digit)
            
            if not digit_dir.exists():
                print(f"  [WARN] Digit {digit} directory not found: {digit_dir}")
                continue
            
            wav_files = list(digit_dir.glob("*.wav"))
            
            for wav_file in wav_files:
                file_info = AudioFileInfo(
                    file_path=wav_file,
                    speaker_id=speaker_id,
                    label=digit  # Label 0-9
                )
                self.file_infos.append(file_info)
                total_added += 1
                per_digit_count[digit] += 1
        
        # Add virtual speaker
        self.speaker_ids.add(speaker_id)
        
        print(f"  [OK] Added {total_added} MyDigits {source_name.upper()} samples as digits 0-9")
        per_digit_str = ', '.join([f'{d}:{per_digit_count[d]}' for d in range(10)])
        print(f"  Per digit: {per_digit_str}")
    
    def _load_noise_samples(self):
        """Load background noise samples for augmentation"""
        noise_files = list(self.noise_dir.glob("*.wav"))
        print(f"Loading {len(noise_files)} noise files for augmentation...")
        
        for nf in noise_files[:20]:  # Limit to 20 for memory efficiency
            try:
                audio, _ = librosa.load(nf, sr=self.SAMPLE_RATE, mono=True)
                self.noise_samples.append(audio.astype(np.float32))
            except Exception as e:
                print(f"[WARN] Failed to load {nf.name}: {e}")
        
        print(f"[OK] Loaded {len(self.noise_samples)} noise samples")
    
    def load_waveform(self, file_path: Path) -> np.ndarray:
        """
        Load audio file as raw waveform (NO external MFCC).
        
        Args:
            file_path: Path to FLAC file
        
        Returns:
            waveform: np.ndarray of shape (16000,), dtype float32, range [-1,1]
        """
        # Load audio with librosa (only for loading, NOT for MFCC)
        waveform, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, mono=True)
        
        # Ensure float32
        waveform = waveform.astype(np.float32)
        
        # Pad or truncate to exactly TARGET_LENGTH
        if len(waveform) < self.TARGET_LENGTH:
            # Pad with zeros
            pad_length = self.TARGET_LENGTH - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), mode='constant', constant_values=0.0)
        elif len(waveform) > self.TARGET_LENGTH:
            # Truncate
            waveform = waveform[:self.TARGET_LENGTH]
        
        # Ensure range [-1, 1] (librosa already returns normalized float)
        # Clip just in case
        waveform = np.clip(waveform, -1.0, 1.0)
        
        return waveform
    
    def apply_time_shift(self, waveform: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply random time-shift augmentation.
        
        Args:
            waveform: Input waveform (16000,)
            seed: Random seed
        
        Returns:
            Time-shifted waveform
        """
        rng = np.random.default_rng(seed)
        shift = rng.integers(-self.TIME_SHIFT_RANGE, self.TIME_SHIFT_RANGE + 1)
        return np.roll(waveform, shift)
    
    def apply_background_noise_mix(self, waveform: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Mix random background noise with the waveform.
        
        Args:
            waveform: Input waveform (16000,)
            seed: Random seed
        
        Returns:
            Waveform with background noise added
        """
        if not self.noise_samples:
            return waveform
        
        rng = np.random.default_rng(seed)
        
        # Pick random noise by index (avoid rng.choice on list of arrays with different lengths)
        noise_idx = rng.integers(0, len(self.noise_samples))
        noise = self.noise_samples[noise_idx]
        
        # Extract random segment
        if len(noise) > self.TARGET_LENGTH:
            start_idx = rng.integers(0, len(noise) - self.TARGET_LENGTH)
            noise_segment = noise[start_idx : start_idx + self.TARGET_LENGTH]
        elif len(noise) == self.TARGET_LENGTH:
            noise_segment = noise
        else:
            # Repeat noise to fill target length
            repeats = (self.TARGET_LENGTH // len(noise)) + 1
            noise_segment = np.tile(noise, repeats)[:self.TARGET_LENGTH]
        
        # Apply random volume
        volume = rng.uniform(*self.NOISE_VOLUME_RANGE)
        mixed = waveform + noise_segment * volume
        
        return mixed
    
    def apply_gain(self, waveform: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply random gain augmentation.
        
        Args:
            waveform: Input waveform (16000,)
            seed: Random seed
        
        Returns:
            Waveform with gain applied
        """
        rng = np.random.default_rng(seed)
        gain = rng.uniform(*self.GAIN_RANGE)
        return waveform * gain
    
    def augment_waveform(self, waveform: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply all augmentations to waveform.
        
        Args:
            waveform: Input waveform (16000,)
            seed: Random seed
        
        Returns:
            Augmented waveform, clipped to [-1,1]
        """
        rng = np.random.default_rng(seed)
        
        # Time-shift
        waveform = self.apply_time_shift(waveform, seed=rng.integers(0, 2**31))
        
        # Background noise mix (50% probability)
        if rng.random() < self.NOISE_MIX_PROB:
            waveform = self.apply_background_noise_mix(waveform, seed=rng.integers(0, 2**31))
        
        # Gain
        waveform = self.apply_gain(waveform, seed=rng.integers(0, 2**31))
        
        # Clip to [-1, 1]
        waveform = np.clip(waveform, -1.0, 1.0)
        
        return waveform
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get count of samples per class"""
        dist = defaultdict(int)
        for info in self.file_infos:
            dist[info.label] += 1
        return dict(dist)
    
    def get_speaker_distribution(self) -> Dict[str, int]:
        """Get count of samples per speaker"""
        dist = defaultdict(int)
        for info in self.file_infos:
            dist[info.speaker_id] += 1
        return dict(dist)
    
    def split_by_speaker(
        self, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1, 
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[AudioFileInfo], List[AudioFileInfo], List[AudioFileInfo]]:
        """
        Split dataset - Phase 4C hybrid approach:
        
        - Digits 0-9: Speaker-based split (NO speaker leakage)
        - _silence_ (10) & _unknown_ (11): Sample-based split (ensures presence in all splits)
        
        This fixes Phase 4 issue where special classes were missing from val/test.
        
        Args:
            train_ratio: Fraction for training (default 0.8)
            val_ratio: Fraction for validation (default 0.1)
            test_ratio: Fraction for testing (default 0.1)
            seed: Random seed for reproducibility
        
        Returns:
            (train_infos, val_infos, test_infos)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        rng = random.Random(seed)
        
        # Separate digit samples (0-9) from special classes (10-11)
        digit_infos = [info for info in self.file_infos if info.label < 10]
        special_infos = [info for info in self.file_infos if info.label >= 10]
        
        # ===== PART 1: Speaker-based split for digits 0-9 =====
        speaker_files: Dict[str, List[AudioFileInfo]] = defaultdict(list)
        for info in digit_infos:
            speaker_files[info.speaker_id].append(info)
        
        # Shuffle speakers deterministically
        speakers = sorted(speaker_files.keys())
        rng.shuffle(speakers)
        
        # Split speakers
        n_speakers = len(speakers)
        n_train = int(n_speakers * train_ratio)
        n_val = int(n_speakers * val_ratio)
        
        train_speakers = speakers[:n_train]
        val_speakers = speakers[n_train:n_train + n_val]
        test_speakers = speakers[n_train + n_val:]
        
        # Collect digit files
        digit_train = [info for spk in train_speakers for info in speaker_files[spk]]
        digit_val = [info for spk in val_speakers for info in speaker_files[spk]]
        digit_test = [info for spk in test_speakers for info in speaker_files[spk]]
        
        # ===== PART 2: Sample-based split for _silence_ (10) & _unknown_ (11) =====
        rng.shuffle(special_infos)
        
        n_special = len(special_infos)
        n_special_train = int(n_special * train_ratio)
        n_special_val = int(n_special * val_ratio)
        
        special_train = special_infos[:n_special_train]
        special_val = special_infos[n_special_train:n_special_train + n_special_val]
        special_test = special_infos[n_special_train + n_special_val:]
        
        # ===== COMBINE =====
        train_infos = digit_train + special_train
        val_infos = digit_val + special_val
        test_infos = digit_test + special_test
        
        # Shuffle combined splits
        rng.shuffle(train_infos)
        rng.shuffle(val_infos)
        rng.shuffle(test_infos)
        
        print(f"\n[OK] Phase 4C Hybrid Split (seed={seed}):")
        print(f"  Digits 0-9: Speaker-based ({len(train_speakers)} train, {len(val_speakers)} val, {len(test_speakers)} test speakers)")
        print(f"  Special classes (_silence_, _unknown_): Sample-based")
        print(f"\n  Train: {len(train_infos)} samples ({len(digit_train)} digits + {len(special_train)} special)")
        print(f"  Val:   {len(val_infos)} samples ({len(digit_val)} digits + {len(special_val)} special)")
        print(f"  Test:  {len(test_infos)} samples ({len(digit_test)} digits + {len(special_test)} special)")
        
        return train_infos, val_infos, test_infos
    
    def create_dataset(
        self, 
        file_infos: List[AudioFileInfo], 
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False,
        seed: Optional[int] = None
    ):
        """
        Create a generator for batching waveforms.
        
        Args:
            file_infos: List of AudioFileInfo to load
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples
            augment: Whether to apply augmentation (Task 5)
            seed: Random seed for shuffling
        
        Yields:
            (batch_X, batch_y): 
                batch_X: (batch_size, 16000) float32
                batch_y: (batch_size, num_classes) one-hot encoded
        """
        rng = random.Random(seed)
        epoch_count = 0
        
        while True:  # Infinite loop for Keras training
            infos = file_infos.copy()
            
            if shuffle:
                rng.shuffle(infos)
            
            num_batches = len(infos) // batch_size
            
            for batch_idx in range(num_batches):
                batch_infos = infos[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                
                batch_X = []
                batch_y = []
                
                for info in batch_infos:
                    # Load raw waveform
                    waveform = self.load_waveform(info.file_path)
                    
                    # Apply augmentation if enabled (training only)
                    if augment:
                        aug_seed = seed + epoch_count * 10000 + batch_idx * batch_size + len(batch_X) if seed else None
                        waveform = self.augment_waveform(waveform, seed=aug_seed)
                    
                    # One-hot encode label
                    label_onehot = np.zeros(self.num_classes, dtype=np.float32)
                    label_onehot[info.label] = 1.0
                    
                    batch_X.append(waveform)
                    batch_y.append(label_onehot)
                
                yield np.array(batch_X), np.array(batch_y)
            
            epoch_count += 1


if __name__ == "__main__":
    # Quick test
    loader = GermanDigitRawWaveformLoader(
        audio_dir="C:/Workspace/Digit_Processing/data/hd_audio/audio_deutsch",
        labels_file="C:/Workspace/Digit_Processing/python/pipeline/optionA_rawwaveform/labels.txt",
        noise_dir="C:/Workspace/Digit_Processing/data/noise"
    )
    
    print("\n=== Class Distribution ===")
    for label, count in sorted(loader.get_class_distribution().items()):
        print(f"  Digit {label}: {count} samples")
    
    print("\n=== Speaker Distribution (first 10) ===")
    for i, (speaker, count) in enumerate(sorted(loader.get_speaker_distribution().items())[:10]):
        print(f"  Speaker {speaker}: {count} samples")
    
    print("\n=== Test waveform loading ===")
    sample_info = loader.file_infos[0]
    waveform = loader.load_waveform(sample_info.file_path)
    print(f"  File: {sample_info.file_path.name}")
    print(f"  Speaker: {sample_info.speaker_id}, Label: {sample_info.label}")
    print(f"  Waveform shape: {waveform.shape}, dtype: {waveform.dtype}")
    print(f"  Range: [{waveform.min():.4f}, {waveform.max():.4f}]")
    
    print("\n=== Test speaker-based split ===")
    train_infos, val_infos, test_infos = loader.split_by_speaker()
    
    print("\n=== Test augmentation ===")
    sample_waveform = loader.load_waveform(sample_info.file_path)
    print(f"  Original range: [{sample_waveform.min():.4f}, {sample_waveform.max():.4f}]")
    
    augmented = loader.augment_waveform(sample_waveform, seed=42)
    print(f"  Augmented range: [{augmented.min():.4f}, {augmented.max():.4f}]")
    print(f"  Augmented shape: {augmented.shape}")
    
    print("\n=== Test batch generation with augmentation ===")
    train_gen = loader.create_dataset(train_infos[:10], batch_size=2, shuffle=False, augment=True, seed=42)
    batch_X, batch_y = next(train_gen)
    print(f"  Batch X shape: {batch_X.shape}, dtype: {batch_X.dtype}")
    print(f"  Batch Y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
    print(f"  Batch X range: [{batch_X.min():.4f}, {batch_X.max():.4f}]")

