import json
from glob import glob

import torch
import math

from dataset import get_data_loader, concat_variable_length_files
from model import FlowGenerator

# def loss(z, m, logs, logdet):
def loss(z, logdet):
  l = 0.5 * torch.sum(z**2) # neg normal likelihood w/o the constant term
#   l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z-m)**2)) # neg normal likelihood w/o the constant term
  l = l - torch.sum(logdet) # log jacobian determinant
  l = l / torch.sum(torch.ones_like(z)) # averaging across batch, channel and time axes
  l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
  return l

if __name__ == "__main__":

    h = json.load(open('config.json', 'r'))

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    speech_files = concat_variable_length_files(sorted(glob('./data/kss/*/*.wav')))
    data_loader = get_data_loader(speech_files, h, num_workers=0)

    print("Before Initialization")

    generator = FlowGenerator(
    #   n_vocab=len(symbols) + getattr(hps.data, "add_blank", False), 
      out_channels=h["num_mels"], 
      **h["model"]).to(device)
    
    # optimizer_g = commons.Adam(generator.parameters(), 
    #                            scheduler=hps.train.scheduler, 
    #                            dim_model=hps.model.hidden_channels, 
    #                            warmup_steps=hps.train.warmup_steps, 
    #                            lr=hps.train.learning_rate, 
    #                            betas=hps.train.betas, 
    #                            eps=hps.train.eps)

    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    step = 0

    for mels in data_loader:
        print(mels.shape)
        # torch.Size([4, 609, 80])
        # torch.Size([4, 80, 609])
        optimizer.zero_grad()
        z, logdet = generator(mels)
        l = loss(z, logdet)
        l.backward()
        optimizer.step()
        # (z, logdet)
        print(z.shape)
        print(logdet.shape)
        print(l.item())
        step += 1

        y, _ = generator(gen=True)

        print(y.shape)

        break