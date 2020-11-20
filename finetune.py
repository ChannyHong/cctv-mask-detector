from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

from PIL import Image, ImageDraw

workers = 0 if os.name == 'nt' else 4


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))



mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device, keep_all=True
)


resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]    

dataset = datasets.ImageFolder('../kaggle-dataset/images_test')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
count = 0
for x, y in loader:
    print(x,y)
    #x_aligned, prob = mtcnn(x, return_prob=True)
    boxes, _ = mtcnn.detect(x)

    image_copy = x.copy()
    draw = ImageDraw.Draw(image_copy)

    print(boxes)
    print(len(boxes))

    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

    print(image_copy)

    image_copy.save("../kaggle-dataset/images_test/all/output{}.jpg".format(count))

    count+=1






