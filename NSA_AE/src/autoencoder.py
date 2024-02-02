#!/usr/bin/python3
import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from .utils import plot_latent_tensorboard, calculate_wasserstein_distance

class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, RTDLoss=None, MSELoss=None, rtd_l=0.1, rtd_every_n_batches=1, rtd_start_epoch=0, lr=5e-4, **kwargs):
        """
        RTDLoss - function of topological (RTD) loss between the latent representation and the input
        l - parameter of regularization lambda (L = L_reconstruct + \lambda L_RTD)
        """
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.decoder = copy.deepcopy(decoder)
        self.norm_constant = nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.RTDLoss = RTDLoss
        self.MSELoss = MSELoss
        self.rtd_l = rtd_l
        self.rtd_every_n_batches = rtd_every_n_batches
        self.rtd_start_epoch = rtd_start_epoch
        self.lr = lr
    
    def forward(self, x):
        embedding = self.norm_constant * self.encoder(x)
        return embedding
    
    def z_dist(self, z):
        z_dist = torch.cdist(z, z)
#         if self.norm_constant is None:
#             self.norm_constant = 1.0 / np.quantile(z_dist.flatten().detach().cpu().numpy(), 0.9)
#         norm_constant = torch.quantile(z_dist.view(-1), 0.9)
        z_dist = self.norm_constant * (z_dist / np.sqrt(z_dist.shape[1]))
        return z_dist

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, x_dist, y = train_batch
        z = self.encoder(x)  
        x_hat = self.decoder(z)
        loss = 0.0
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('train/mse_loss', loss)
        if self.RTDLoss is not None:
            if (self.rtd_start_epoch <= self.current_epoch) and batch_idx % self.rtd_every_n_batches == 0:
                z_dist = self.z_dist(z)
                loss_xz, loss_zx, rtd_loss = self.RTDLoss(x_dist, z_dist)
                self.log('train/rtd_loss', rtd_loss)
                self.log('train/rtd_loss_xz', loss_xz)
                self.log('train/rtd_loss_zx', loss_zx)
                loss += self.rtd_l*rtd_loss
        self.log('train/loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_dist, y = val_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = 0.0
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('val/mse_loss', loss)
        if self.RTDLoss is not None and self.rtd_start_epoch <= self.current_epoch+1:
            z_dist = self.z_dist(z)
            loss_xz, loss_zx, rtd_loss = self.RTDLoss(x_dist, z_dist)
            self.log('val/rtd_loss', rtd_loss)
            self.log('val/rtd_loss_xz', loss_xz)
            self.log('val/rtd_loss_zx', loss_zx)
            loss += self.rtd_l*rtd_loss
        self.log('val/loss', loss)

class NSAAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, NSALoss=None, MSELoss=None, nsa_l=0.1, nsa_every_n_batches=1, nsa_start_epoch=0, lr=5e-4, **kwargs):
        """
           NSALoss - minimize pairwise discrepancy
        """
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.decoder = copy.deepcopy(decoder)
        self.norm_constant = nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.NSALoss = NSALoss
        self.MSELoss = MSELoss
        self.nsa_l = nsa_l
        self.nsa_every_n_batches = nsa_every_n_batches
        self.nsa_start_epoch = nsa_start_epoch
        self.lr = lr
    
    def forward(self, x):
        embedding = self.norm_constant * self.encoder(x)
        return embedding
    
    def z_dist(self, z):
        z_dist = torch.cdist(z, z)
        norm_value = 2*torch.quantile(torch.sqrt(torch.sum(z**2, axis=1)),0.9)
        z_dist = self.norm_constant * (z_dist / norm_value)
        return z_dist

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, x_dist, y = train_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        #x_dist = x_dist / np.quantile(x_dist.flatten().detach().cpu().numpy(),0.9)
        #x_dist = x_dist / 2*np.quantile(np.sqrt(np.sum(x.detach().cpu().numpy()**2,axis=1)),0.9)
        #x_dist = x_dist / 2*torch.quantile(torch.sqrt(torch.sum(x**2, axis=1)),0.9)
        loss = 0.0
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('train/mse_loss', loss)
        if self.NSALoss is not None:
            if (self.nsa_start_epoch <= self.current_epoch) and batch_idx % self.nsa_every_n_batches == 0:
                #z_dist = self.z_dist(z)
                loss_nsa = self.NSALoss(x, z)
                loss += self.nsa_l*loss_nsa
        self.log('train/loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_dist, y = val_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = 0.0
        #x_dist = x_dist / 2*np.quantile(np.sqrt(np.sum(x.detach().cpu().numpy()**2,axis=1)),0.9)
        #x_dist = x_dist / 2*torch.quantile(torch.sqrt(torch.sum(x**2, axis=1)),0.9)
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('val/mse_loss', loss)
        if self.NSALoss is not None and self.nsa_start_epoch <= self.current_epoch+1:
            # z_dist = self.z_dist(z)
            # z_dist = self.z_dist(z)
            loss_nsa = self.NSALoss(x, z)
            loss += self.nsa_l*loss_nsa
        self.log('val/loss', loss)


class NSAAutoEncoder_2(pl.LightningModule):
    def __init__(self, encoder, decoder, LIDNSALoss = None, NSALoss=None, MSELoss=None, nsa_l=0.1, nsa_every_n_batches=1, nsa_start_epoch=0, lid_nsa_l=0.1, lid_nsa_every_n_batches=1, lid_nsa_start_epoch=0, lr=5e-4, **kwargs):
        """
           NSALoss - minimize pairwise discrepancy
        """
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.decoder = copy.deepcopy(decoder)
        self.norm_constant = nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.NSALoss = NSALoss
        self.LIDNSALoss = LIDNSALoss
        self.MSELoss = MSELoss
        self.nsa_l = nsa_l
        self.nsa_every_n_batches = nsa_every_n_batches
        self.nsa_start_epoch = nsa_start_epoch
        self.lid_nsa_l = lid_nsa_l
        self.lid_nsa_every_n_batches = lid_nsa_every_n_batches
        self.lid_nsa_start_epoch = lid_nsa_start_epoch
        self.lr = lr

    def forward(self, x):
        embedding = self.norm_constant * self.encoder(x)
        return embedding

    def z_dist(self, z):
        z_dist = torch.cdist(z, z)
        norm_value = 2*torch.quantile(torch.sqrt(torch.sum(z**2, axis=1)),0.9)
        z_dist = self.norm_constant * (z_dist / norm_value)
        return z_dist

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, x_dist, y = train_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        #x_dist = x_dist / np.quantile(x_dist.flatten().detach().cpu().numpy(),0.9)
        #x_dist = x_dist / 2*np.quantile(np.sqrt(np.sum(x.detach().cpu().numpy()**2,axis=1)),0.9)
        #x_dist = x_dist / 2*torch.quantile(torch.sqrt(torch.sum(x**2, axis=1)),0.9)
        loss = 0.0
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('train/mse_loss', loss)
        if self.NSALoss is not None:
            if (self.nsa_start_epoch <= self.current_epoch) and batch_idx % self.nsa_every_n_batches == 0:
                #z_dist = self.z_dist(z)
                loss_nsa = self.NSALoss(x, z)
                loss += self.nsa_l*loss_nsa
        if self.LIDNSALoss is not None:
            if (self.lid_nsa_start_epoch <= self.current_epoch) and batch_idx % self.lid_nsa_every_n_batches == 0:
                #z_dist = self.z_dist(z)
                loss_lid_nsa = self.LIDNSALoss(x, z)
                loss += self.lid_nsa_l*loss_lid_nsa
        self.log('train/loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, x_dist, y = val_batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = 0.0
        #x_dist = x_dist / 2*np.quantile(np.sqrt(np.sum(x.detach().cpu().numpy()**2,axis=1)),0.9)
        #x_dist = x_dist / 2*torch.quantile(torch.sqrt(torch.sum(x**2, axis=1)),0.9)
        if self.MSELoss is not None:
            loss += self.MSELoss(x_hat, x)
            self.log('val/mse_loss', loss)
        if self.NSALoss is not None and self.nsa_start_epoch <= self.current_epoch+1:
            # z_dist = self.z_dist(z)
            # z_dist = self.z_dist(z)
            loss_nsa = self.NSALoss(x, z)
            loss += self.nsa_l*loss_nsa
        if self.LIDNSALoss is not None and self.lid_nsa_start_epoch <= self.current_epoch+1:
            # z_dist = self.z_dist(z)
            # z_dist = self.z_dist(z)
            loss_lid_nsa = self.LIDNSALoss(x, z)
            loss += self.nsa_l*loss_lid_nsa
        self.log('val/loss', loss)