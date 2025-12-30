# GuitarVision ML Data

This directory contains datasets for training and evaluating the guitar fingering recognition model.

## Directory Structure

```
data/
├── raw/              # Original captured data (tracked with DVC)
│   ├── chord_C/
│   ├── chord_G/
│   ├── chord_D/
│   └── ...
├── processed/        # Processed features ready for training
│   ├── features.npy
│   ├── labels.npy
│   └── metadata.json
└── README.md        # This file
```

## Data Collection Strategy

### Capture Protocol
- **Device**: Webcam (720p minimum)
- **Lighting**: Well-lit, consistent conditions
- **Hand visibility**: Both hands visible, fingers clear
- **Samples per chord**: 200-300 images minimum
- **Variability**: Multiple angles, slight position shifts

### Chord Labels
Starting with open chords:
- C, D, E, G, A
- Am, Em, Dm

Progressive additions:
- Barre chords (F, Bm)
- 7th chords (G7, D7, A7)
- Extended chords (as model improves)

## Data Versioning

All datasets in `raw/` and `processed/` are tracked with DVC:

```bash
# Track new data
dvc add ml/data/raw/chord_C/

# Push to remote storage
dvc push

# Pull latest data
dvc pull
```

## Dataset Metadata

Each capture session generates:
- Images (`.jpg`)
- MediaPipe landmarks (`.json`)
- Session metadata (`session_info.json`)

## Quality Checks

Before training:
- ✅ All images have detected hands
- ✅ Landmark quality score > 0.8
- ✅ Balanced samples across chords
- ✅ No duplicate frames
