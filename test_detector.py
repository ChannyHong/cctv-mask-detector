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



import random
import os

from mtcnn import MTCNN
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--detector_model_path', type=str, help='Required: the path to the detector .pt model file', required=True)
parser.add_argument('--test_examples_path', type=str, help='Required: the path to the folder consisting of "protected" and "unprotected" test examples', default="mask_dataset/test")
parser.add_argument('--mtcnn_model_path', type=str, help='Optional: the path to the custom MTCNN .pt model file', default=None)

args = parser.parse_args()

class Detector(nn.Module):
    def __init__(self, pretrained_model_path):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.mp = nn.MaxPool2d(4)
        self.fc = nn.Linear(640, 1)

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

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device, keep_all=True
    )

    detector = Detector(pretrained_model_path=args.detector_model_path)

    test_image_dir = args.test_examples_path

    y_pred = []
    y_data = []

    protected_files = os.listdir(os.path.join(test_image_dir, "protected"))
    protected_files.remove(".DS_Store")

    unprotected_files = os.listdir(os.path.join(test_image_dir, "unprotected"))
    unprotected_files.remove(".DS_Store")

    # LOAD TEST EXAMPLES
    test_examples = []

    for protected_file in protected_files:
        protected_image = Image.open(os.path.join(test_image_dir, "protected", protected_file))
        protected_faces = mtcnn(protected_image)

        if protected_faces is not None:
            # multiple protected faces in the image
            if protected_faces.size()[0] > 1:
                for protected_face in protected_faces:
                    test_examples.append((protected_face, 1))
            # one protected face in the image
            else:
                protected_face = torch.squeeze(protected_faces)
                test_examples.append((protected_face, 1))

    for unprotected_file in unprotected_files:
        unprotected_image = Image.open(os.path.join(test_image_dir, "unprotected", unprotected_file))
        unprotected_faces = mtcnn(unprotected_image)
        
        if unprotected_faces is not None:
            # multiple protected faces in the image
            if unprotected_faces.size()[0] > 1:
                for unprotected_face in unprotected_faces:
                    test_examples.append((unprotected_face, 0))
            # one protected face in the image
            else:
                unprotected_face = torch.squeeze(unprotected_faces)
                test_examples.append((unprotected_face, 0))
    

    # SHUFFLE TEST EXAMPLES
    random.shuffle(test_examples)

    # CHECK THE ANSWERS
    num_true_positive = 0
    num_false_positive = 0

    num_true_negative = 0
    num_false_negative = 0

    for image, label in test_examples:
        image = [image]
        image = torch.stack(image)
        image = torch.squeeze(image, 1)

        pred = detector(image)

        # prediction = positive (protected)
        if pred >= 0.5:
            if label == 1:
                num_true_positive += 1
            elif label == 0:
                num_false_positive += 1

        # prediction = negative (unprotected)
        elif pred < 0.5:
            if label == 0:
                num_true_negative += 1
            elif label == 1:
                num_false_negative += 1

    # PRINT RESULTS
    print("All done!")
    print("num_true_positive: ", num_true_positive)
    print("num_false_positive: ", num_false_positive)
    print("num_true_negative: ", num_true_negative)
    print("num_false_negative: ", num_false_negative)

    num_correct = num_true_positive + num_true_negative
    num_incorrect = num_false_positive + num_false_negative

    print("num_correct: ", num_correct)
    print("num_incorrect: ", num_incorrect)

    print("Accuracy: ", float(num_correct) / float(num_correct + num_incorrect))

if __name__ == "__main__":
    main()
