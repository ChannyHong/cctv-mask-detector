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

import os
import numpy as np

from mtcnn import MTCNN
from PIL import Image, ImageDraw
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--footage_path', type=str, help='Required: the path to the footage file', default="mask_dataset/test")
parser.add_argument('--detector_model_path', type=str, help='Required: the path to the detector .pt model file', required=True)
parser.add_argument('--output_path', type=str, help='Required: the path to output the annotated video file to', required=True)
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

    if args.mtcnn_model_path:
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.1, 0.2, 0.2], factor=0.709, post_process=True,
            device=device, keep_all=True, pretrained_model_path=args.mtcnn_model_path
        )
    else:
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.1, 0.2, 0.2], factor=0.709, post_process=True,
            device=device, keep_all=True
        )

    detector = Detector(pretrained_model_path=args.detector_model_path)

    vidcap = cv2.VideoCapture(args.footage_path)
    success, frame = vidcap.read() # get the first frame

    height, width, _ = frame.shape

    # Deconstruct the video and load onto memory by reading one by one

    annotated_frames = []

    while success:

        # Convert cv2 image to PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        image_annotated = image.copy()
        draw = ImageDraw.Draw(image_annotated)

        face_box_pairs = mtcnn.detect_face_box_pairs(image)

        if face_box_pairs is not None:
            for face_box_pair in face_box_pairs:
                face, box = face_box_pair

                face = [face]
                face = torch.stack(face)
                face = torch.squeeze(face, 1)

                pred = detector(face)

                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)
                # draw label here too

                if pred >= 0.5:
                    draw.text((box[0], box[1]-15), "protected", (255, 255, 255))

                elif pred < 0.5:
                    draw.text((box[0], box[1]-15), "unprotected", (255, 255, 255))

        annotated_frames.append(image_annotated)

        # get the next frame
        success, frame = vidcap.read()



    # WRITE THE VIDEO NOW
    #video = cv2.VideoWriter(args.output_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    video = cv2.VideoWriter(args.output_path, -1, 12, (width, height))

    for annotated_frame in annotated_frames:

        # Convert PIL Image back to cv2 image
        upload_frame = np.array(annotated_frame)

        # Convert RGB to BGR 
        upload_frame = cv2.cvtColor(np.asarray(upload_frame), cv2.COLOR_RGB2BGR)

        video.write(upload_frame)

    video.release()






if __name__ == "__main__":
    main()
