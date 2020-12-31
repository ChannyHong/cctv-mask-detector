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

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_images_dir', type=str, help='Required: the path to the folder consisting of "protected" and "unprotected" train examples', default="mask_dataset/train")
parser.add_argument('--train_metadata_dir', type=str, help='Required: the path to the folder consisting of "protected" and "unprotected" train examples', default="mask_dataset/train")
parser.add_argument('--train_labels_dir', type=str, help='Required: the path to the folder consisting of the training data labels', default="mask_dataset/train")
parser.add_argument('--batch_size', type=int, help='Required: the number of train examples to include in a batch per class', default=8)
parser.add_argument('--num_epochs', type=int, help='Required: the number of epochs to run training', default=50)
parser.add_argument('--detector_model_output_path', type=str, help='The path to save the trained detector model file', required=True)
parser.add_argument('--mtcnn_model_path', type=str, help='Optional: the path to the custom MTCNN .pt model file', default=None)
parser.add_argument('--pretrained_detector_model_path', type=str, help='Optional: the path to the custom detector .pt model file', default=None)


args = parser.parse_args()



BATCH_SIZE = args.batch_size



class Detector(nn.Module):
    def __init__(self, pretrained_model_path=None):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.mp = nn.MaxPool2d(4)
        self.fc = nn.Linear(640, 1)

        if pretrained_model_path:
            state_dict = torch.load(pretrained_model_path)
            self.load_state_dict(state_dict)

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

    etector = Detector(pretrained_model_path=args.detector_model_path)


    train_image_list = os.listdir(args.train_metadata_dir)
    train_dataset_size = len(train_image_list)



    criterion = nn.BCELoss() # binary cross entropy

    optimizer = torch.optim.Adam(detector.parameters(), lr=0.001)

    # Training loop
    for epoch in range(args.num_epochs):

        # calculate how many iterations in the epoch
        iterations = int(train_dataset_size/BATCH_SIZE)

        # Train through every batch
        for iteration_num in range(iterations):
            x_data = []
            y_data = []
            y_pred = []

            batch_start_index = iteration_num*BATCH_SIZE

            for i in range(batch_start_index, batch_start_index+BATCH_SIZE):
                metadata_filename = train_image_list[i]
                metadata_file_pointer = open(os.path.join(args.train_metadata_dir, metadata_filename), "r+")
                metadata_dict = json.load(metadata_file_pointer)
                metadata_file_pointer.close()
                label_id = metadata_dict["label_id"]

                image_filename = metadata_filename[:-5] + ".jpg"
                image = Image.open(os.path.join(args.train_images_dir, image_filename))

                label_filename = "{}.json".format(label_id)
                label_file_pointer = open(os.path.join(args.train_labels_dir, label_filename), "r+")
                label_dict = json.load(label_file_pointer)
                label_file_pointer.close()
                label_objects = label_dict["result"]["objects"]

                # NOTE: If we have more than 1 face in the image, we are not going to use it to finetune the mask detector, for simplicity purposes
                if len(label_objects) > 1:
                    continue
                else:
                    classification = label_objects[0]["class"] # assuming we only have 
                    if classification = "protected":
                        y_data.append(torch.tensor(1.0))
                    elif classification = "unprotected":
                        y_data.append(torch.tensor(0.0))

                faces = mtcnn(image)
                if faces is not None:
                    # NOTE: Again, for simplicity purposes, we are only going to use the face is a finetuning example if we only have one face recognized by our MTCNN
                    if protected_faces.size()[0] == 1:
                        protected_face = torch.squeeze(protected_faces)
                        x_data.append(protected_face)

            x_data = torch.stack(x_data)
            x_data = torch.squeeze(x_data, 1)

            y_pred = detector(x_data)

            y_data = torch.stack(y_data)
            y_data = torch.unsqueeze(y_data, 1)

            loss = criterion(y_pred, y_data)

            print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch, iteration_num, loss))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(detector.state_dict(), args.detector_model_output_path)


if __name__ == "__main__":
    main()
