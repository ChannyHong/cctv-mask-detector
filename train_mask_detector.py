# Copyright 2020 Superb AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Channy Hong


import json
import os

from mtcnn import MTCNN
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchviz import make_dot

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_examples_path', type=str, help='Required: the path to the folder consisting of "protected" and "unprotected" train examples', default="mask_dataset/train")
parser.add_argument('--train_dataset_size_per_class', type=int, help='Required: the number of train examples in "protected" and "unprotected" folders (or the smaller of the two)', default=500)
parser.add_argument('--batch_size_per_class', type=int, help='Required: the number of train examples to include in a batch per class', default=4)
parser.add_argument('--num_epochs', type=int, help='Required: the number of epochs to run training', default=50)
parser.add_argument('--detector_model_output_path', type=str, help='The path to save the trained detector model file', default="models/detector.pt")
parser.add_argument('--mtcnn_model_path', type=str, help='Optional: the path to the custom MTCNN .pt model file', default=None)



args = parser.parse_args()



BATCH_SIZE_PER_CLASS = args.batch_size_per_class



class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.mp = nn.MaxPool2d(4)
        self.fc = nn.Linear(640, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return torch.sigmoid(x)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    print('Running on device: {}'.format(device))


    if args.mtcnn_model_path:
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device, keep_all=True, pretrained_model_path=args.mtcnn_model_path
        )
    else:
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device, keep_all=True
        )

    detector = Detector()

    train_image_dir = args.train_examples_path
    train_dataset_size = args.train_dataset_size_per_class

    criterion = nn.BCELoss() # binary cross entropy

    optimizer = torch.optim.Adam(detector.parameters(), lr=0.001)

    # Training loop
    for epoch in range(args.num_epochs):

        # calculate how many iterations in the epoch
        iterations = int(train_dataset_size/BATCH_SIZE_PER_CLASS)

        # Train through every batch
        for iteration_num in range(iterations):
            x_data = []
            y_data = []
            y_pred = []

            batch_start_index = iteration_num*BATCH_SIZE_PER_CLASS

            for i in range(batch_start_index, batch_start_index+BATCH_SIZE_PER_CLASS):
                protected_image = Image.open(os.path.join(train_image_dir, "protected", str(i+1)+".jpeg"))
                protected_faces = mtcnn(protected_image)

                if protected_faces is not None:
                    # multiple protected faces in the image
                    if protected_faces.size()[0] > 1:
                        for protected_face in protected_faces:
                            x_data.append(protected_face)
                            y_data.append(torch.tensor(1.0))
                    # one protected face in the image
                    else:
                        protected_face = torch.squeeze(protected_faces)
                        x_data.append(protected_face)
                        y_data.append(torch.tensor(1.0))

                unprotected_image = Image.open(os.path.join(train_image_dir, "unprotected", str(i+1)+".jpeg"))
                unprotected_faces = mtcnn(unprotected_image)

                if unprotected_faces is not None:
                    # multiple unprotected faces in the image
                    if unprotected_faces.size()[0] > 1:
                        for unprotected_face in unprotected_faces:
                            x_data.append(unprotected_face)
                            y_data.append(torch.tensor(0.0))
                    # one unprotected face in the image
                    else:
                        unprotected_face = torch.squeeze(unprotected_faces)
                        x_data.append(unprotected_face)
                        y_data.append(torch.tensor(0.0))

            x_data = torch.stack(x_data)
            x_data = torch.squeeze(x_data, 1)

            y_pred = detector(x_data)

            y_data = torch.stack(y_data)
            y_data = torch.unsqueeze(y_data, 1)

            loss = criterion(y_pred, y_data)

            #model_arch = make_dot(loss)
            #model_arch.format = 'png'
            #model_arch.render("loss_graphs/epoch{}-iter{}-loss.png".format(epoch, iteration_num))

            print("y_pred: ", y_pred)
            print("y_data: ", y_data)

            print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch, iteration_num, loss))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if loss < 0.1:
            #    torch.save(detector.state_dict(), "./test_ep{}_iter{}_loss{}.pt".format(epoch, iteration_num, loss))

    torch.save(detector.state_dict(), args.detector_model_output_path)


if __name__ == "__main__":
    main()
