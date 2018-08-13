"""
This file includes some functions for visualization in jupyter notebook
"""

import torch
import torch.utils.data as data
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
import os


def get_loader(root, files, transform, batch_size, shuffle, num_workers):

    dataset = ImageExtractDataset(root, files, transform)
    data_loader = torch.utils.data.DataLoader( dataset=dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers,
                                               collate_fn=dataset.collate_fn)

    return data_loader

class ImageExtractDataset(data.Dataset):

    def __init__(self, root, files=None, transform=None):
        self._root = root
        self._transform = transform
        self._files = self._collect_images() if files is None else files

    def _collect_images(self):
        root = self._root
        files = [f for f in listdir(root)]
        return files

    def __getitem__(self, index):
        root = self._root
        files = self._files
        image = join(root, files[index])
        image = Image.open(image).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        if self._transform is not None:
            image = self._transform(image)

        return image, files[index]

    def __len__(self):
        return len(self._files)

    def collate_fn(self, data):
        images, names = zip(*data)
        images = torch.stack(images, 0)

        return images, names


__all__ = ['decode_captions',
           'attention_visualization']


def decode_captions(captions, idx_to_word):
    N, D = captions.shape
    # print("N, D", N, D)
    decoded = []
    for idx in range(N):
        words = []
        for wid in range(D):
            # print(idx, wid, captions[idx, wid].data, captions[idx, wid].item())
            word = idx_to_word[captions[idx, wid].item()]
            if word == '<end>' or word == '<start>' or word == '<unk>':
                words.append('.')
            else:
                words.append(word)
        decoded.append(words)
    return decoded


def attention_visualization(root, image_name, caption, alphas):
    image = Image.open(os.path.join(root, image_name))
    image = image.resize([224, 224], Image.LANCZOS)
    plt.subplot(4, 5, 1)

    plt.imshow(image)
    plt.axis('off')

    words = caption[1:]
    for t in range(len(words)):
        if t > 18:
            break
        plt.subplot(4, 5, t + 2)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=10)
        plt.imshow(image)
        # print alphas
        alp_curr = alphas[t, :].view(14, 14)
        alp_img = skimage.transform.pyramid_expand(alp_curr.numpy(), upscale=16, sigma=20)
        plt.imshow(alp_img, alpha=0.85)
        plt.axis('off')


    plt.show()



