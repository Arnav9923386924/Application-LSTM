# Music Generation with LSTM

## Overview
This project implements a deep learning-based melody generation model using an LSTM neural network. The model is trained on musical data in **Kern** format and can generate melodies based on a seed sequence.

## Features
- **Preprocessing**: Converts musical scores into encoded sequences.
- **LSTM Model**: Uses an LSTM neural network to learn patterns from music.
- **Melody Generation**: Generates music sequences given an initial seed melody.
- **MIDI Export**: Saves generated melodies as MIDI files.

## Dependencies
Make sure to install the required Python libraries before running the code:
```bash
pip install torch pandas numpy matplotlib music21 scikit-learn
```

## Dataset
The dataset used for training consists of **Kern** files stored in the specified `DATASET_PATH`. These files are parsed using **music21** and converted into numerical sequences.

## Usage
### 1. Preprocess Data
Run the preprocessing script to prepare training data:
```python
preprocess(DATASET_PATH)
```

### 2. Train the Model
Train the LSTM model using:
```python
train()
```

### 3. Generate Music
To generate a melody from a seed sequence:
```python
mg = MelodyGenerator()
seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
melody = mg.generate_melody(seed, 1000, SEQUENCE_LENGTH, 1.5)
print(melody)
mg.save_melody(melody)
```
This will save the melody as a MIDI file.

## Model Architecture
- **Input:** One-hot encoded musical sequences
- **LSTM Layer:** Processes sequences and learns temporal dependencies
- **Dropout Layer:** Prevents overfitting
- **Fully Connected Layer:** Outputs probability distribution over possible next notes
- **Softmax Activation:** Generates the next note based on probability

## Hyperparameters
- **Sequence Length**: 64
- **LSTM Units**: 256
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 50

## File Structure
```
.
├── music_dataset/        # Preprocessed dataset
├── model.pth            # Trained model weights
├── mapping.json         # Mapping of notes to integers
├── file_dataset         # Encoded dataset as a single file
└── generated.mid        # Example generated melody
```

## Future Improvements
- Implement Transformer-based melody generation
- Improve dataset with more diverse musical styles
- Add GUI for interactive melody generation

---
Happy composing! 🎶

