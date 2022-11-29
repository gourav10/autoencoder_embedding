import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from model import Encoder, Decoder


class Engine:
    def __init__(self,
                 encoded_space_dim,
                 num_epochs=config.NUM_EPOCHS) -> None:
        if (torch.cuda.is_available()):
            self.device = "cuda"
        else:
            self.device = "cpu"
        torch.manual_seed(0)
        self.encoder = Encoder(encoded_space_dim=encoded_space_dim,
                               fc2_input_dim=128)
        self.decoder = Decoder(encoded_space_dim=encoded_space_dim)
        self.num_epochs = num_epochs

    def get_autoencoder(self):
        return self.encoder, self.decoder

    def __train(self, dataloader, loss_fn, optimizer):
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(self.device)
            encoded_data = self.encoder(image_batch)
            decoded_data = self.decoder(encoded_data)
            loss = loss_fn(decoded_data, image_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def __validate(self, dataloader, loss_fn):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            conc_out = []
            conc_label = []
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(self.device)
                encoded_data = self.encoder(image_batch)
                decoded_data = self.decoder(encoded_data)
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            val_loss = loss_fn(conc_out, conc_label)
        return val_loss.item()

    def train_model(self, train_dl, test_dl, loss_fn, optimizer):
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        history = {'train_loss': [], 'val_loss': []}
        min_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = self.__train(train_dl, loss_fn, optimizer)
            val_loss = self.__validate(test_dl, loss_fn)

            if (epoch % 5 == 0):
                print('\n EPOCH {}/{} \t train loss{:.3f} \t val loss {:.3f}'
                      .format(epoch + 1, self.num_epochs,
                              train_loss, val_loss))

            if (epoch % 10 == 0):
                self.save_model(model_name=f'model_{epoch}')

            if (min_loss > val_loss):
                min_loss = val_loss
                self.save_model(model_name='best')

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
        return history

    def save_model(self, model_name=''):
        os.makedirs('temp', exist_ok=True)
        dest_path_encoder = os.path.join('temp', f'{model_name}_encoder.pth')
        dest_path_decoder = os.path.join('temp', f'{model_name}_decoder.pth')
        torch.save(self.encoder, dest_path_encoder)
        torch.save(self.decoder, dest_path_decoder)

    def plot_train_loss(self, history):
        plt.figure(figsize=(10, 8))
        plt.semilogy(history['train_loss'], label='Train')
        plt.semilogy(history['val_loss'], label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.show()
