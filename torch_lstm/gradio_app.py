import gradio as gr
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
import uuid

# 1. Device (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Redefine your LSTMClassifier (exactly as in training)
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 bidirectional=True, dropout=0.5):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * (2 if bidirectional else 1),
                                 num_classes)

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

# 3. Instantiate & load your trained weights
input_size  = 13 + 12   # MFCC + Chroma dims
hidden_size = 128       # same as training
num_layers  = 2
num_classes = 8         # RAVDESS has 8 emotion labels

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load('torch_lstm/best_lstm_ser.pth', map_location=device))
model.to(device)
model.eval()

# 4. Emotion label mapping
# (Use exactly the same mapping you trained with!)
emotion2idx = {'neutral':0,'calm':1,'happy':2,'sad':3,'angry':4,'fearful':5,'disgust':6,'surprised':7}
idx2emotion = {v:k for k,v in emotion2idx.items()}
emotion_names = [idx2emotion[i] for i in range(num_classes)]


def predict(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    feats  = np.vstack([mfcc, chroma]).T            # shape (T, D)
    feats  = torch.from_numpy(feats).float().unsqueeze(0)  # (1, T, D)
    length = torch.tensor([feats.size(1)])

    # 6. Predict & get probabilities
    with torch.no_grad():
        logits = model(feats.to(device), length.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    # 7. Plot confidence across emotions
    plt.figure(figsize=(8,4))
    plt.bar(emotion_names, probs)
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Model Confidence for Your Sample')
    plt.xticks(rotation=30)
    plt.tight_layout()
    # plt.savefig('emotion_confidence.png')
    f_name = f'gradio_results/{uuid.uuid4()}.png'
    plt.savefig(f_name)
    return f_name

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=['upload', 'microphone'], type='filepath', label="Audio Sample"),
    outputs=gr.Image(label='Model Confidence'),
)

demo.launch(share=True)