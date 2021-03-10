# https://github.com/jaywalnut310/glow-tts

import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import commons


class FlowSpecDecoder(nn.Module):
  def __init__(self, 
      in_channels, 
      hidden_channels, 
      kernel_size, 
      dilation_rate, 
      n_blocks, 
      n_layers, 
      p_dropout=0., 
      n_split=4,
      n_sqz=2,
      sigmoid_scale=False,
      gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_blocks = n_blocks
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for b in range(n_blocks):
      self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
      self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
      self.flows.append(
        modules.CouplingBlock(
          in_channels * n_sqz,
          hidden_channels,
          kernel_size=kernel_size, 
          dilation_rate=dilation_rate,
          n_layers=n_layers,
          gin_channels=gin_channels,
          p_dropout=p_dropout,
          sigmoid_scale=sigmoid_scale))

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      flows = self.flows
      logdet_tot = 0
    else:
      flows = reversed(self.flows)
      logdet_tot = None

    if self.n_sqz > 1:
      x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)
    for f in flows:
      if not reverse:
        # x, logdet = f(x, x_mask, g=g, reverse=reverse)
        x, logdet = f(x, x_mask, reverse=reverse)
        logdet_tot += logdet
      else:
        # x, logdet = f(x, x_mask, g=g, reverse=reverse)
        x, logdet = f(x, x_mask, reverse=reverse)
    if self.n_sqz > 1:
      x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)
    return x, logdet_tot

  def store_inverse(self):
    for f in self.flows:
      f.store_inverse()


class FlowGenerator(nn.Module):
  def __init__(self, 
      hidden_channels, 
      filter_channels, 
      # filter_channels_dp, 
      out_channels,
      kernel_size=3, 
      n_heads=2, 
      n_layers_enc=6,
      p_dropout=0., 
      n_blocks_dec=12, 
      kernel_size_dec=5, 
      dilation_rate=5, 
      n_block_layers=4,
      p_dropout_dec=0., 
      n_speakers=0, 
      gin_channels=0, 
      n_split=4,
      n_sqz=1,
      sigmoid_scale=False,
      window_size=None,
      block_length=None,
      mean_only=False,
      hidden_channels_enc=None,
      hidden_channels_dec=None,
      prenet=False,
      n_vocab=0, 
      **kwargs):

    super().__init__()
    # self.n_vocab = n_vocab
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    # self.filter_channels_dp = filter_channels_dp
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_heads = n_heads
    self.n_layers_enc = n_layers_enc
    self.p_dropout = p_dropout
    self.n_blocks_dec = n_blocks_dec
    self.kernel_size_dec = kernel_size_dec
    self.dilation_rate = dilation_rate
    self.n_block_layers = n_block_layers
    self.p_dropout_dec = p_dropout_dec
    # self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.hidden_channels_enc = hidden_channels_enc
    self.hidden_channels_dec = hidden_channels_dec

    self.decoder = FlowSpecDecoder(
        out_channels, 
        hidden_channels_dec or hidden_channels, 
        kernel_size_dec, 
        dilation_rate, 
        n_blocks_dec, 
        n_block_layers, 
        p_dropout=p_dropout_dec, 
        n_split=n_split,
        n_sqz=n_sqz,
        sigmoid_scale=sigmoid_scale,
        gin_channels=gin_channels)

  def forward(self, y=None, y_lengths=None, gen=False, noise_scale=1., length_scale=1.):

    x, x_lengths = None, None
    g=None,

    # x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)

    if gen:
      # w = torch.exp(logw) * x_mask * length_scale
      # w_ceil = torch.ceil(w)
      # y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
      y_max_length = 256
      y_lengths = torch.ones(4, dtype=torch.int64) * y_max_length
    else:
      # y_lengths = torch.arange(max_length, dtype=length.dtype, device=length.device)
      y_max_length = y.size(2)
      y_lengths = torch.ones(y.size(0), dtype=torch.int64) * y_max_length
      y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
    # z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1)
    # attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

    if gen:
      # attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
      # z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      # z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      # logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

      # z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
      z = torch.randn((4, 80, y_max_length))
      y, logdet = self.decoder(z, z_mask, g=g, reverse=True)
      print('GEN!')
      print(y)
      print(logdet)
      # return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
      # return (y, logdet), (x_m, x_logs, x_mask), (attn, logw, logw_)
      return y, logdet
    else:
      # z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
      z, logdet = self.decoder(y, z_mask, reverse=False)
      with torch.no_grad():
        pass
        # x_s_sq_r = torch.exp(-2 * x_logs)
        # logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
        # logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
        # logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
        # logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
        # logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

        # attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
      # z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      # z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      # logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
      # return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
      # return (z, logdet, z_mask)
      return z, logdet

  def preprocess(self, y, y_lengths, y_max_length):
    if y_max_length is not None:
      y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
      y = y[:,:,:y_max_length]
    y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
    return y, y_lengths, y_max_length

  def store_inverse(self):
    self.decoder.store_inverse()
