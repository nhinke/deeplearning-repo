import torch

class Encoder(torch.nn.Module):

    def __init__(self, num_input_channels: int=3, num_middle_channels: int=8, num_latent_channels: int=8, kernel_size: int=3, stride: int=1, padding: int=0):
        
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.num_input_channels = num_input_channels
        self.num_middle_channels = num_middle_channels
        self.num_latent_channels = num_latent_channels

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.num_input_channels, out_channels=self.num_middle_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.num_middle_channels, out_channels=self.num_latent_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            torch.nn.ReLU()
        )

    def forward(self, input):
        return self.cnn(input)


class Decoder(torch.nn.Module):

    def __init__(self, num_input_channels: int=3, num_middle_channels: int=8, num_latent_channels: int=8, kernel_size: int=3, stride: int=1, padding: int=0):

        super().__init__()

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.num_input_channels = num_input_channels
        self.num_middle_channels = num_middle_channels
        self.num_latent_channels = num_latent_channels

        self.cnn = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=self.num_latent_channels, out_channels=self.num_middle_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=self.num_middle_channels, out_channels=self.num_input_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.cnn(input)


class AutoEncoder(torch.nn.Module):

    def __init__(self, sgd_optim: bool=True, learning_rate: float=1e-3, momentum: float=0.9, weight_decay: float=1e-5, num_input_channels: int=3, num_middle_channels: int=8, num_latent_channels: int=8, kernel_size: int=3, stride: int=1, padding: int=0, model_path='autoencoder.pth'):

        super().__init__()

        self.momentum = momentum
        self.sgd_optim = sgd_optim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.num_input_channels = num_input_channels
        self.num_middle_channels = num_middle_channels
        self.num_latent_channels = num_latent_channels

        self.model_path = model_path
        self.encoder = Encoder(num_input_channels=self.num_input_channels, num_middle_channels=self.num_middle_channels, num_latent_channels=self.num_latent_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.decoder = Decoder(num_input_channels=self.num_input_channels, num_middle_channels=self.num_middle_channels, num_latent_channels=self.num_latent_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.criterion = torch.nn.MSELoss()
        if (self.sgd_optim):
            self.optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def forward(self, x):
        latent_rep = self.encoder(x)
        xhat = self.decoder(latent_rep)
        return xhat

    # 'inputs' and 'true_inputs' are the same for regular autoencoder, but 'inputs' are noisy inputs for denoising autoencoder
    def training_step(self, inputs, true_inputs):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, true_inputs)
        loss.backward()
        self.optimizer.step()
        return outputs, loss

    # 'inputs' and 'true_inputs' are the same for regular autoencoder, but 'inputs' are noisy inputs for denoising autoencoder
    def validation_step(self, inputs, true_inputs):
        with torch.no_grad():
            outputs = self(inputs)
            loss = self.criterion(outputs, true_inputs)
        return outputs, loss
        
    def load_saved_model(self):
        print(f'Loading model parameters from: {self.model_path}\n')
        return self.load_state_dict(state_dict=torch.load(self.model_path))

    def save_model(self):
        print(f'Saving model parameters to: {self.model_path}\n')
        torch.save(self.state_dict(),self.model_path)
