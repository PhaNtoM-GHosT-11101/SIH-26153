import torch
import torch.nn as nn
import math


class LSTMWorldModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        logits = self.classifier(last_hidden)
        attention_weights = torch.softmax(logits, dim=-1)
        return logits, attention_weights


class TransformerWorldModel(nn.Module):
    def __init__(self, input_size: int, d_model: int = 128, num_heads: int = 8,
                 num_layers: int = 2, num_classes: int = 6, dropout: float = 0.2,
                 max_seq_len: int = 100):
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        transformer_out = self.transformer(x)
        last_hidden = transformer_out[:, -1, :]
        logits = self.classifier(last_hidden)

        attn = torch.softmax(logits, dim=-1)
        return logits, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GRUWorldModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        logits = self.classifier(last_hidden)
        attn = torch.softmax(logits, dim=-1)
        return logits, attn


def build_world_model(model_type: str, input_size: int, num_classes: int = 6, **kwargs) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "lstm":
        return LSTMWorldModel(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == "transformer":
        return TransformerWorldModel(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == "gru":
        return GRUWorldModel(input_size=input_size, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from lstm, transformer, gru")
