import os
import glob

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

modality = ['full-av', 'video-only', 'audio-only']
vocal_channel = ['speech', 'song']
emotion = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
intensity = ['normal', 'strong']

data_dir = '../data'
wav_paths = glob.glob(os.path.join(data_dir, 'Actor_*', '*.wav'))

rows = []
for p in wav_paths:
    fname = os.path.basename(p)
    # ravdess goes
    # modality - channel - emotion - intensity - statement - repetition - actor
    parts = fname.replace('.wav', '').split('-')
    rows.append({
        'path': p,
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': emotion[int(parts[2])-1],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6]
    })

meta = pd.DataFrame(rows)
print(f"Total samples: {len(meta)}")
print("Emotions present:", sorted(meta['emotion'].unique()))
print("Counts by emotion:")
print(meta['emotion'].value_counts())

meta.head()

def get_duration(path):
    return librosa.get_duration(path=path)

meta['duration'] = meta['path'].map(get_duration)

meta.describe()

# get path with smallest duration
print(meta.loc[meta['duration'].idxmin()])

def get_mfcc(path):
    n_mfcc = 13
    y, sr = librosa.load(path, sr=16000)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

mfcc_shapes = []

for path in meta['path']:
    mfcc = get_mfcc(path)
    mfcc_shapes.append(mfcc.shape)

mfcc_shapes = pd.DataFrame(mfcc_shapes)
mfcc_shapes.describe()

def get_chroma(path):
    n_mfcc = 13
    y, sr = librosa.load(path, sr=16000)
    return librosa.feature.chroma_stft(y=y, sr=sr)

chroma_shapes = []

for path in meta['path']:
    chroma = get_chroma(path)
    chroma_shapes.append(chroma.shape)

chroma_shapes = pd.DataFrame(chroma_shapes)
chroma_shapes.describe()

# for speed: sample a few files per emotion
grouped = meta.groupby('emotion').sample(5, random_state=0)

mfcc_stats = []
for _, row in grouped.iterrows():
    y, sr = librosa.load(row['path'], sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_stats.append({
        'emotion': row['emotion'],
        **{f'mfcc{i+1}_mean': mfcc[i].mean() for i in range(13)}
    })
mfcc_stats_df = pd.DataFrame(mfcc_stats)

# boxplot of, say, mfcc1_mean by emotion
plt.figure(figsize=(8,4))
mfcc_stats_df.boxplot(column='mfcc1_mean', by='emotion')
plt.title('Mean MFCC1 by Emotion')
plt.suptitle('')
plt.ylabel('MFCC1')
plt.show()

# for speed: sample a few files per emotion
grouped = meta.groupby('emotion').sample(5, random_state=0)

chroma_stats = []
for _, row in grouped.iterrows():
    y, sr = librosa.load(row['path'], sr=None)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    chroma_stats.append({
        'emotion': row['emotion'],
        **{f'chroma{i+1}_mean': chroma[i].mean() for i in range(12)}
    })
chroma_stats_df = pd.DataFrame(chroma_stats)

# boxplot of, say, mfcc1_mean by emotion
plt.figure(figsize=(8,4))
chroma_stats_df.boxplot(column='chroma3_mean', by='emotion')
plt.title('Mean Chroma1 by Emotion')
plt.suptitle('')
plt.ylabel('Chroma3')
plt.show()

# DATA LOADER
emotions = sorted(meta['emotion'].unique())
emotion2idx = {emo: i for i, emo in enumerate(emotions)}

from tqdm import tqdm  # for progress bar
import torch

# Hyper-params (must match what you use later)
SR = None
N_MFCC = 13
N_CHROMA = 12

# Lists to hold your tensors and labels
feats_cache = []
labels_cache = []

print("Pre-computing features…")
for _, row in tqdm(meta.iterrows(), total=len(meta)):
    y, sr = librosa.load(row['path'], sr=SR)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    feats  = np.vstack([mfcc, chroma]).T            # (frames, feat_dim)
    tensor = torch.from_numpy(feats).float()
    label  = torch.tensor(emotion2idx[row['emotion']], dtype=torch.long)
    feats_cache.append(tensor)
    labels_cache.append(label)
    # print(f"mfcc: {mfcc.shape}, chroma: {chroma.shape}, label: {row['emotion']}")

# Optionally save to disk for later sessions:
torch.save((feats_cache, labels_cache), "ravdess_feats.pt")
print("Done! Cached", len(feats_cache), "samples.")

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class CachedRAVDESSDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats  = feats
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]

from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    """
    batch: list of (feat_tensor [T_i×D], label)
    returns:
      - feats: (B, T_max, D) padded with zeros
      - lengths: (B,) original T_i
      - labels: (B,)
    """
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    # pad to the max T in this batch
    feats_padded = pad_sequence(feats, batch_first=True)  # zeros out the shorter ones
    labels = torch.stack(labels)
    return feats_padded, lengths, labels

from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader

feats_cache, labels_cache = torch.load("ravdess_feats.pt")

# Instantiate dataset & DataLoader
dataset = CachedRAVDESSDataset(feats_cache, labels_cache)

train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

# loader  = DataLoader(
#     dataset,
#     batch_size=32,
#     shuffle=True,
#     collate_fn=pad_collate,
#     num_workers=2,
#     pin_memory=False
# )

train_ds, val_ds = random_split(dataset,
                                [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))


train_loader = DataLoader(train_ds,
                          batch_size=32,
                          shuffle=True,
                          collate_fn=pad_collate,
                          num_workers=2)

val_loader   = DataLoader(val_ds,
                          batch_size=32,
                          shuffle=False,
                          collate_fn=pad_collate,
                          num_workers=2)

for feats, lengths, labels in train_loader:
    print("Train loader")
    print("feats:", feats.shape)    # e.g. (32, T_max, 25)
    print("lengths:", lengths)      # e.g. tensor([200, 180, 240, …])
    print("labels:", labels.shape)  # (32,)
    break

for feats, lengths, labels in val_loader:
    print("Test loader")
    print("feats:", feats.shape)    # e.g. (32, T_max, 25)
    print("lengths:", lengths)      # e.g. tensor([200, 180, 240, …])
    print("labels:", labels.shape)  # (32,)
    break

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 bidirectional: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bidirectional=bidirectional,
                      batch_first=True,
                      dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)

        if self.lstm.bidirectional:
            n_layers = self.lstm.num_layers
            H = self.lstm.hidden_size
            h_n = h_n.view(n_layers, 2, -1, H)
            h_fwd = h_n[-1, 0]
            h_bwd = h_n[-1, 1]
            h = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h = h_n[-1]

        out = self.dropout(h)
        logits = self.fc(out)
        return logits

import torch.optim as optim

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0.0, 0
    for feats, lengths, labels in loader:
        feats, lengths, labels = feats.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(feats, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for feats, lengths, labels in loader:
            feats, lengths, labels = feats.to(device), lengths.to(device), labels.to(device)
            logits = model(feats, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy

# 3.1 Hyper-parameters & setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size   = 13 + 12             # MFCCs + Chroma
hidden_size  = 128
num_layers   = 2
num_classes  = len(emotion2idx)
lr           = 1e-3
batch_size   = 32
num_epochs   = 100

# 3.2 Instantiate
model     = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# optional scheduler:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='max',
                                                 patience=3,
                                                 factor=0.5)

# 3.3 Training loop
best_val_acc = 0.0
for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss,   val_acc   = eval_epoch(model, val_loader, criterion, device)  # or a separate val_loader

    log_line = (f"Epoch {epoch:02d}  "
            f"Train: loss={train_loss:.3f}, acc={train_acc:.3f}  "
            f"Val:   loss={val_loss:.3f}, acc={val_acc:.3f}")
    print(log_line)
    
    with open("training_log.txt", "a") as f:
        f.write(log_line + "\n")


    # step scheduler on val_acc
    scheduler.step(val_acc)

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_lstm_emotion_run_2.pth")

print(f"Best val acc: {best_val_acc:.3f}")
