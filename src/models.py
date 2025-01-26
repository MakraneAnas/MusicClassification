import torch.nn as nn


class MusicCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3, device="cuda"):
        super(MusicCNN, self).__init__()
        self.device = device

        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        ).to(device)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        ).to(device)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        ).to(device)

        self.fc_layers = None  # Fully connected layers will be initialized later
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Flatten dynamically
        x = x.view(x.size(0), -1)

        # Initialize FC layers dynamically
        if self.fc_layers is None:
            fc_input_size = x.size(1)
            self.fc_layers = nn.Sequential(
                nn.Linear(fc_input_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, self.num_classes)
            ).to(self.device)

        x = self.fc_layers(x)
        return x


class MusicCRNN2D(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1, gru_hidden_size=32, device="cuda"):
        super(MusicCRNN2D, self).__init__()
        self.device = device

        # Input batch normalization
        self.input_bn = nn.BatchNorm2d(1).to(device)

        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout_rate)
        ).to(device)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(dropout_rate)
        ).to(device)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(dropout_rate)
        ).to(device)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((4, 2)),
            nn.Dropout2d(dropout_rate)
        ).to(device)

        self.gru_stack = None  # GRU layers will be initialized later
        self.classifier = None
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.gru_hidden_size = gru_hidden_size

    def forward(self, x):
        x = self.input_bn(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Reshape for GRU
        batch_size, _, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, time, -1)

        # Initialize GRU dynamically
        if self.gru_stack is None:
            gru_input_size = x.size(2)
            self.gru_stack = nn.GRU(
                input_size=gru_input_size,
                hidden_size=self.gru_hidden_size,
                batch_first=True,
                bidirectional=True,
            ).to(self.device)
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate * 3),
                nn.Linear(self.gru_hidden_size * 2, self.num_classes)  # * 2 for bidirectional
            ).to(self.device)

        x, _ = self.gru_stack(x)

        # Take the last time step
        x = x[:, -1, :]

        # Classification
        x = self.classifier(x)
        return x


class MusicCRNN1D(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2, gru_hidden_size=64, device="cuda"):
        super(MusicCRNN1D, self).__init__()
        self.device = device

        self.first_conv_initialized = False
        self.gru_initialized = False
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.gru_hidden_size = gru_hidden_size

        # These will be initialized in the first forward pass
        self.conv_block1 = None
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        ).to(device)

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        ).to(device)

        self.gru1 = None
        self.gru2 = None
        self.classifier = None

    def _initialize_first_conv(self, input_channels):
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate)
        ).to(self.device)
        self.first_conv_initialized = True

    def _initialize_gru(self, input_size):
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=self.gru_hidden_size,
            batch_first=True
        ).to(self.device)

        self.gru2 = nn.GRU(
            input_size=self.gru_hidden_size,
            hidden_size=self.gru_hidden_size,
            batch_first=True
        ).to(self.device)

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.gru_hidden_size, self.num_classes)
        ).to(self.device)

        self.gru_initialized = True

    def forward(self, x):
        # x shape: (batch, channels, frequency, time)
        batch_size = x.size(0)

        # Permute and reshape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, frequency)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, channels*frequency)

        # Initialize first conv layer if needed
        if not self.first_conv_initialized:
            self._initialize_first_conv(x.size(2))

        # Apply conv blocks
        x = x.transpose(1, 2)  # (batch, channels*frequency, time)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Prepare for GRU
        x = x.transpose(1, 2)  # (batch, time, channels)

        # Initialize GRU if needed
        if not self.gru_initialized:
            self._initialize_gru(x.size(2))

        # Apply GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Take the output from the last time step
        x = x[:, -1, :]  # (batch, hidden_size)

        # Classification
        x = self.classifier(x)
        return x


class MusicRNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2, lstm_hidden_size=64, device="cuda"):
        super(MusicRNN, self).__init__()
        self.device = device

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lstm_hidden_size = lstm_hidden_size

        # Initialize with a small dummy input size that will be replaced
        self.lstm1 = nn.LSTM(
            input_size=1,  # Will be replaced in forward pass
            hidden_size=self.lstm_hidden_size,
            batch_first=True
        ).to(self.device)

        self.lstm2 = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            batch_first=True
        ).to(self.device)

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.lstm_hidden_size, self.num_classes)
        ).to(self.device)

        self.initialized = False

    def _reinitialize_lstm(self, input_size):
        """Reinitialize the first LSTM layer with correct input size"""
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=self.lstm_hidden_size,
            batch_first=True
        ).to(self.device)
        self.initialized = True

    def forward(self, x):
        # x shape: (batch, channels, frequency, time)
        batch_size = x.size(0)

        # Permute and reshape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, frequency)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, channels*frequency)

        # Reinitialize LSTM if needed
        if not self.initialized:
            self._reinitialize_lstm(x.size(2))

        # Apply LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Take the output from the last time step
        x = x[:, -1, :]  # (batch, hidden_size)

        # Classification
        x = self.classifier(x)
        return x