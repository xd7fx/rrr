import torch

class EmotionRecognizer(torch.nn.Module):
    def __init__(self, num_classes: int = 5, hidden_size: int = 384):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.classifier(h_n[-1])
