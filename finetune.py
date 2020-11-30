from torch import nn
import torch
from torch import tensor

from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw



BATCH_SIZE = 8



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device, keep_all=True
)



print(mtcnn)
for param in mtcnn.parameters():
	print(param)


def iou(box1, box2): # box = [top_left_x, top_left_y, width, height]
	




def criterion(y_pred, y_data):
	loss = 0.0

	for y_hat, y_gt in zip(y_pred, y_data):
		
		bd_regression_losses = []

		for gt_box in y_gt:
			
			
			closest_prediction = TODO

			bd_regression_losses.append(TODO)



	# identify how 'off' the predictions were on the following factors
	# 1. area of the unaccounted box, compared to the average area of size of the ground truths
	# 2. Euclidean distance of the center point of the box to the closest ground truth box 



	return loss




dataset_list = os.listdir(DATA_PATH)
dataset_size = len(dataset_list)


#criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(mtcnn.parameters(), lr=0.01)


# Training loop
for epoch in range(500):

	# start looking from the first image
	iterations = dataset_size / BATCH_SIZE

	for iteration in iterations:

		x_data = []
		y_data = []
		batch_start_index = iteration*BATCH_SIZE
		for i in range(batch_start_index, batch_start_index+BATCH_SIZE):
			image = Image.TODO
			x_data.append(image)
			y_data.append(TODO)

		y_pred, _ = mtcnn.detect(x_data) # batch size, 4
		loss = criterion(y_pred, y_data)

		print(f'Epoch: {epoch} | Loss: {loss.item()} ')

	    # Zero gradients, perform a backward pass, and update the weights.
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()






class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(mtcnn.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())


