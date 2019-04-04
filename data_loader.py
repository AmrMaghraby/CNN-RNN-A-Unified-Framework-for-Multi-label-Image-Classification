import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
    def __len__(self):
        return len(self.ids)

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root, json=json,vocab=vocab,transform=transform)
    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco,batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers,collate_fn=collate_fn)
    return data_loader