import json
import random
import os

from mtcnn import MTCNN
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchviz import make_dot


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
        print(x)
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.sigmoid(x)


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device, keep_all=True
    )

    detector = Detector(pretrained_model_path="models/test_end.pt")

    test_image_dir = "mask_dataset/test"

    y_pred = []
    y_data = []

    protected_files = os.listdir(os.path.join(test_image_dir, "protected"))
    #protected_files.remove(".DS_Store")

    unprotected_files = os.listdir(os.path.join(test_image_dir, "unprotected"))
    #unprotected_files.remove(".DS_Store")


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
        print(image)
        print(label)
        pred = detector(image)

        # prediction = positive (protected)
        if pred >= 0.5:
            if label == 1:
                num_true_positive += 1
            elif label == 0:
                num_false_positive += 1

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
    num_incorrect = num_true_positive + num_true_negative

    print("num_correct: ", num_correct)
    print("num_incorrect: ", num_incorrect)

    print("Accuracy: ", float(num_correct) / float(num_correct + num_incorrect))

        




if __name__ == "__main__":
    main()
