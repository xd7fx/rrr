import torch

class EmotionRecognizer(torch.nn.Module):
    def __init__(self, num_classes: int = 5, hidden_size: int = 384):
        super().__init__()
        print("Initializing EmotionRecognizer...")
        self.hidden_size = hidden_size
        self.rnn = torch.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        print("GRU initialized successfully!")
        self.classifier = torch.nn.Linear(hidden_size, num_classes)
        print("Classifier initialized successfully!")

    def forward(self, x):
        print("Forward pass started...")
        _, h_n = self.rnn(x)
        print("GRU forward pass completed!")
        return self.classifier(h_n[-1])
