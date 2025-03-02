import torch 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import music21 as m21
import matplotlib.image as mpimg
import json
from sklearn.preprocessing import OneHotEncoder
SAVE_DIR = "music_dataset"
DATASET_PATH = r"D:\deutschl\essen\europa\deutschl\erk"
SINGLE_FILE_DATASET = "file_dataset" 
SEQUENCE_LENGTH = 64
MAPPING_PATH = "mapping.json"
dataset = sorted(os.listdir(DATASET_PATH))
print("Loaded categories:", dataset)
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
# parse means to read, analyze and breakdown data into structural format 
def load_songs_in_kern(DATASET_PATH):
    songs = []
    for path, subdirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    
    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    
    return True

def transpose(song): 
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    
    # estimate key using music21 if the key is not noted 
    if not isinstance(key, m21.key.Key): 
        key = song.analyze("key")
    
    print(key)
    # get the interval for transposition. eg., Bmaj -> Cmaj 
    if key.mode == "major": 
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    
    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    
    return transposed_song 

def encode_song(song, time_step = 0.25): 
    encoded_song = []
    # the end representation would be like [60, "_", "_", "_"]
    for event in song.flatten().notesAndRests: 
        if isinstance(event, m21.note.Note): 
            symbol = event.pitch.midi # here 60
            print(symbol)
        # case to handle rest
        if isinstance(event, m21.note.Rest): 
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps): 
            if step == 0: 
                encoded_song.append(symbol)
            else: 
                encoded_song.append("_")
    encoded_song = " ".join(map(str, encoded_song))
    
    return encoded_song

def load(file_path): 
    with open(file_path, "r") as f: 
        song = f.read()
    return song
    
def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    # remove empty space from last character of string
    songs = songs[:-1]
    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

def create_mapping(songs, mapping_path): 
    mappings = {}
    # indentify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    # create mappings
    for i, symbol in enumerate(vocabulary): 
        mappings[symbol] = i
    # save the vocabulary to a json file
    with open (mapping_path, "w") as f: 
        json.dump(mappings, f)

def convert_songs_to_int(songs): 
    int_song = []
    # load the mappings
    with open(MAPPING_PATH, "r") as f: 
        mappings = json.load(f)
    
    # cast songs string to list 
    songs = songs.split()
    
    # map songs to int
    for symbol in songs: 
        int_song.append(mappings[symbol])

    return int_song

def  generate_training_sequences(sequence_length): 
    # load the songs and map them to int   
    inputs = []
    targets = []
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    # generate the training sequences 
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences): 
        inputs.append(int_songs[i: i + sequence_length])
        targets.append(int_songs[i + sequence_length])

    # input size: [number of sequences, sequence length] -> after one-hot encoding-> 
    # [number pf sequence, sequence length, vocabulary size] 
    
    # one-hot encode the sequences
    inputs = np.array(inputs)
    vocabulary_size = len(set(int_songs))  
    encoder = OneHotEncoder(sparse_output = False)
    inputs_flattened = inputs.reshape(-1, 1)
    encoded_flattened = encoder.fit_transform(inputs_flattened)
    encoded_inputs = encoded_flattened.reshape(num_sequences, sequence_length, vocabulary_size)
    targets = np.array(targets)
    return encoded_inputs, targets
def preprocess(DATASET_PATH): 
    print("Loading songs...")
    songs = load_songs_in_kern(DATASET_PATH)
    print(f"Loaded {len(songs)} songs")

    for i, song in enumerate(songs): 
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS): 
            continue
        song = transpose(song)
        encoded_song = encode_song(song)
 
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as f: 
            f.write(encoded_song)
preprocess(DATASET_PATH)
songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET,SEQUENCE_LENGTH) 
mapping = create_mapping(songs, MAPPING_PATH)
# print(set(int_songs_list))
inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
# print(inputs)
# print(targets)
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available else "cpu" 
device
OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.pth"
def build_model(output_units, num_units):
    class MelodyModel(nn.Module):
        def __init__(self, output_units, num_units):
            super(MelodyModel, self).__init__()
            self.lstm = nn.LSTM(output_units, num_units[0], batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(num_units[0], output_units)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out

    return MelodyModel(output_units, num_units)

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss_fn=LOSS, learning_rate=LEARNING_RATE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    inputs, targets = torch.tensor(inputs, dtype=torch.float32).to(device), torch.tensor(targets, dtype=torch.long).to(device)
    
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = build_model(output_units, num_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()
class MelodyGenerator:
    def __init__(self, model_path="model.pth"): 
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model().to(self.device)
        with open(MAPPING_PATH, "r") as f: 
            self._mappings = json.load(f)
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
        self.encoder = OneHotEncoder(sparse_output=False, categories=[range(len(self._mappings))])

    def _load_model(self):
        model = build_model(output_units=38, num_units=[256])
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature): 
        seed = seed.split()
        melody = seed 
        seed = self._start_symbols + seed

        # Convert seed symbols to integer indices
        seed = [self._mappings[symbol] for symbol in seed]
       
        for _ in range(num_steps): 
            seed = seed[-max_sequence_length:]

            # Perform one-hot encoding
            seed_array = np.array(seed).reshape(-1, 1)
            one_hot_seed = self.encoder.fit_transform(seed_array)
            one_hot_seed = torch.tensor(one_hot_seed, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Model prediction
            with torch.inference_mode():
                predictions = self.model(one_hot_seed)
                predictions = predictions.cpu().numpy().flatten()

            # Apply temperature scaling
            predictions = np.exp(predictions / temperature) / np.sum(np.exp(predictions / temperature))

            # Sample next symbol
            next_index = np.random.choice(len(predictions), p=predictions)
            next_symbol = list(self._mappings.keys())[list(self._mappings.values()).index(next_index)]

            melody.append(next_symbol)
            seed.append(next_index)

        return " ".join(melody)

    def save_melody(self, melody, step_duration=0.5, format="midi", file_name="mel.mid"):
        stream = m21.stream.Stream()
        start_symbol = None
        step_counter = 1  
    
        for i, symbol in enumerate(melody.split()):
            if symbol != "_" and symbol.strip(): 
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
    
                    if start_symbol == "r":  
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    elif start_symbol.isdigit():  
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    else:
                        continue  
    
                    stream.append(m21_event)
                    step_counter = 1  
                start_symbol = symbol  
            else:
                step_counter += 1
    
        if start_symbol and start_symbol.strip():
            quarter_length_duration = step_duration * step_counter
            if start_symbol == "r":
                m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
            elif start_symbol.isdigit():
                m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
            else:
                m21_event = None
    
            if m21_event:
                stream.append(m21_event)

        stream.write(format, file_name)
mg = MelodyGenerator()
seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
melody = mg.generate_melody(seed, 1000, SEQUENCE_LENGTH, 1.5)
print(melody)
mg.save_melody(melody)
