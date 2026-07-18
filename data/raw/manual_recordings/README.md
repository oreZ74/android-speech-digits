# Manual Recordings (MyDigits)

Custom recordings for domain adaptation. Not committed to this repository.

Record using the app's debug function. Files are saved under `mydigits_android/<digit>/`.

```
manual_recordings/
└── mydigits_android/
    ├── 0/   # WAV files per digit
    ├── 1/
    └── ...
```

Format: WAV, any sample rate (resampled to 16 kHz), ~1 s per utterance. Aim for ≥10 recordings per digit.

After recording:

```bash
python -m digit_pipeline.scripts.register_mydigits
```
