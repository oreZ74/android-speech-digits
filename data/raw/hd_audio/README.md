# Heidelberg Digits (HD Audio)

German spoken digits 0–9. Not included in this repository — download manually.

**Source:** [Zenke Lab](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

## Dataset Stats

| Metric              | Value                                        |
| ------------------- | -------------------------------------------- |
| Total files         | 5,330                                        |
| Speakers            | 12 (IDs 0–11)                                |
| Digits              | 10 (0–9)                                     |
| Samples per speaker | 190–650 (~44 per digit per speaker)          |
| Samples per digit   | 533                                          |
| Format              | FLAC, 48 kHz (original), resampled to 16 kHz |

## Expected Structure

```
hd_audio/
├── audio_deutsch/          # FLAC files go here
│   └── lang-german_speaker-XX_trial-YY_digit-Z.flac
├── train_filenames.txt
└── test_filenames.txt
```
