import json
import os

from mtcnn import MTCNN
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchviz import make_dot

BATCH_SIZE_PER_CLASS = 4



class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.mp = nn.MaxPool2d(4)
        self.fc = nn.Linear(640, 1)

        #self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        #self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        #self.fc = nn.Linear(102400, 1)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        #print("x: ", x)
        #print("F.log_softmax(x): ", F.log_softmax(x))
        return F.sigmoid(x)
        #x = self.block1(x)
        #x = self.maxpool(x)
        #x = self.block2(x)
        #x = self.block3(x)
        #x = self.maxpool(x)
        #x = x.view(in_size, -1) # flatten the tensor
        #x = self.fc(x)
        #return F.sigmoid(x)

    def conv_block(self, c_in, c_out, dropout, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block


def normalize(tti, itt, tensor):
    return itt(tti(tensor))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device, keep_all=True#, pretrained_model_path="test.pt"
    )

    detector = Detector()

    image_to_tensor = transforms.ToTensor()
    tensor_to_image = transforms.ToPILImage(mode="RGB")

    train_image_dir = "mask_dataset/train"
    train_dataset_size = 500

    criterion = nn.BCELoss() # binary cross entropy

    optimizer = torch.optim.Adam(detector.parameters(), lr=0.001)


    # Training loop
    for epoch in range(50):

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
                            #protected_face = normalize(tensor_to_image, image_to_tensor, protected_face)
                            x_data.append(protected_face)
                            y_data.append(torch.tensor(1.0))
                    # one protected face in the image
                    else:
                        protected_face = torch.squeeze(protected_faces)
                        #protected_face = normalize(tensor_to_image, image_to_tensor, protected_face)
                        x_data.append(protected_face)
                        y_data.append(torch.tensor(1.0))

                unprotected_image = Image.open(os.path.join(train_image_dir, "unprotected", str(i+1)+".jpeg"))
                unprotected_faces = mtcnn(unprotected_image)

                if unprotected_faces is not None:
                    # multiple unprotected faces in the image
                    if unprotected_faces.size()[0] > 1:
                        for unprotected_face in unprotected_faces:
                            #unprotected_face = normalize(tensor_to_image, image_to_tensor, unprotected_face)
                            x_data.append(unprotected_face)
                            y_data.append(torch.tensor(0.0))
                    # one unprotected face in the image
                    else:
                        unprotected_face = torch.squeeze(unprotected_faces)
                        #unprotected_face = normalize(tensor_to_image, image_to_tensor, unprotected_face)
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

    torch.save(detector.state_dict(), "models/test_end.pt")


if __name__ == "__main__":
    main()
