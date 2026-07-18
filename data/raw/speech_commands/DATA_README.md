# Google Speech Commands Dataset

## Quelle
TensorFlow Speech Commands Dataset v0.02
https://www.tensorflow.org/datasets/catalog/speech_commands

## Zweck
- Training der `_unknown_` Klasse (nicht-Ziffern)
- Background Noise für Data Augmentation

## Struktur (nach Download)
```
speech_commands/
├── _background_noise_/     # Hintergrundgeräusche für Augmentation
│   ├── doing_the_dishes.wav
│   ├── dude_miaowing.wav
│   ├── exercise_bike.wav
│   ├── pink_noise.wav
│   ├── running_tap.wav
│   └── white_noise.wav
├── zero/                   # Englische Wörter (für _unknown_)
├── one/
├── two/
├── yes/
├── no/
├── up/
├── down/
├── left/
├── right/
├── on/
├── off/
├── stop/
├── go/
└── ...
├── testing_list.txt
├── validation_list.txt
├── LICENSE
└── README.md
```

## Download
```bash
# Option 1: TensorFlow Datasets
pip install tensorflow-datasets
python -c "import tensorflow_datasets as tfds; tfds.load('speech_commands')"

# Option 2: Direkter Download
# https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
```

## Verwendung im Training
Der DataLoader nutzt automatisch:
- `_background_noise_/` für Augmentation
- Nicht-Ziffern-Ordner (yes, no, up, etc.) für `_unknown_` Klasse
