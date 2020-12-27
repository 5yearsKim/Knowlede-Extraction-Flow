from dataloader import prepare_data
from torchvision.utils import save_image
import os

from KEflow.config import TYPE_DATA

_, dataset = prepare_data('./data', TYPE_DATA)

if not os.path.exists(f'aided_sample/{TYPE_DATA}'):
    os.makedirs(f'aided_sample/{TYPE_DATA}')

for i in range(500):
    x, label = dataset[i]
    save_image(x, os.path.join("aided_sample", TYPE_DATA, f"{i}_image{label}.png"), normalize=True)
