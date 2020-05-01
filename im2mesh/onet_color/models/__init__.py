import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.onet_color.models import encoder_latent, decoder

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
    'cbatchnorm2': decoder.DecoderCBatchNorm2,
    'batchnorm': decoder.DecoderBatchNorm,
    'cbatchnorm_noresnet': decoder.DecoderCBatchNormNoResnet,
}


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, encoderG=None, encoderC=None, encoder_latent=None, p0_z=None,
                 device=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder.to(device)

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        if encoderG is not None:
            self.encoderG = encoderG.to(device)
        else:
            self.encoderG = None

        if encoderC is not None:
            self.encoderC = encoderC.to(device)
        else:
            self.encoderC = None

        self._device = device
        self.p0_z = p0_z

    def forward(self, p, input_g, input_c, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        c0 = self.encode_input_g(input_g)
        c1 = self.encode_input_c(input_c)
        z = self.get_z_from_prior((batch_size,), sample=sample)
        p_r = self.decode(p, z, torch.cat([c0, c1], dim=1), **kwargs)
        return p_r

    def compute_elbo(self, p, colors, input_g, input_c, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''
        c0 = self.encode_input_g(input_g)
        c1 = self.encode_input_c(input_c)
        q_z = self.infer_z(p, colors, torch.cat((c0, c1), dim=1), **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, torch.cat((c0, c1), dim=1), **kwargs)

        rec_error = torch.abs(colors - p_r).mean()
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def encode_input_g(self, input_g):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoderG is not None:
            c = self.encoderG(input_g)
        else:
            # Return inputs?
            c = torch.empty(input_g.size(0), 0)

        return c

    def encode_input_c(self, input_c):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoderC is not None:
            c = self.encoderC(input_c)
        else:
            # Return inputs?
            c = torch.empty(input_c.size(0), 0)

        return c

    def decode(self, p, z, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, z, c, **kwargs)
        p_r = nn.Tanh()(logits)
        return p_r

    def infer_z(self, p, color, c, **kwargs):
        ''' Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, color, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
