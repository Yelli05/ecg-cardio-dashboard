import torch.nn as nn

class ECGCNN2(nn.Module):
    def __init__(self, num_classes=5, channels=(16, 32, 64), use_bn=False, dropout=0.3, hidden=64):
        super().__init__()
        layers, in_ch, length = [], 1, 187
        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers += [nn.ReLU(), nn.MaxPool1d(2)]
            in_ch, length = out_ch, length // 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * length, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))