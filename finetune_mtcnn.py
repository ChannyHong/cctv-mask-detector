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

import torch
from torch import nn, tensor
from torch.autograd import Variable

import json
import os
from mtcnn import MTCNN
from PIL import Image, ImageDraw

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_dir', type=str, help='Required: the path to the folder consisting of label metadata files', required=True)
parser.add_argument('--labels_dir', type=str, help='Required: the path to the folder consisting of label files', required=True)
parser.add_argument('--images_dir', type=str, help='Required: the path to the folder consisting of the images', required=True)

parser.add_argument('--num_epochs', type=int, help='Required: the number of epochs to run training', required=True)
parser.add_argument('--batch_size', type=int, help='Required: the number of images to include in a batch', required=True)

parser.add_argument('--mtcnn_model_output_path', type=str, help='Required: the path to output the finetuned MTCNN model', required=True)
parser.add_argument('--pretrained_mtcnn_model', type=str, help='Optional: the path to the custom MTCNN .pt model to start from', default=None)


args = parser.parse_args()



BATCH_SIZE = args.batch_size

def iou(gt_box, pred_box): # gt_box = [top_left_x, top_left_y, width, height], pred_box = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

	# Determine x, y coordinates of the two boxes
	box1_x1 = torch.tensor(float(gt_box["x"]), requires_grad=True)
	box1_x2 = torch.tensor(float(gt_box["x"]) + gt_box["width"], requires_grad=True)
	box1_y1 = torch.tensor(float(gt_box["y"]), requires_grad=True)
	box1_y2 = torch.tensor(float(gt_box["y"]) + gt_box["height"], requires_grad=True)

	box2_x1 = pred_box[0]
	box2_x2 = pred_box[2]
	box2_y1 = pred_box[1]
	box2_y2 = pred_box[3]

	# determine the (x, y)-coordinates of the intersection rectangle
	#xA = torch.max(torch.tensor([box1_x1, box2_x1]))
	xA = torch.max(box1_x1, box2_x1)
	yA = torch.max(box1_y1, box2_y1)
	xB = torch.min(box1_x2, box2_x2)
	yB = torch.min(box1_y2, box2_y2)

	# compute the area of intersection rectangle
	interArea = torch.mul(torch.max(torch.tensor(0), torch.sub(xB, xA)), torch.max(torch.tensor(0), torch.sub(yB, yA)))

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = torch.mul(torch.sub(box1_x2, box1_x1), torch.sub(box1_y2, box1_y1))
	boxBArea = torch.mul(torch.sub(box2_x2, box2_x1), torch.sub(box2_y2, box2_y1))

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = torch.div(interArea, torch.sub(torch.add(boxAArea, boxBArea), interArea))
	return iou

def dist(gt_box, pred_box): # gt_box = [top_left_x, top_left_y, width, height], pred_box = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]	

	# Determine x, y coordinates of the two boxes
	box1_x1 = torch.tensor(float(gt_box["x"]), requires_grad=True)
	box1_x2 = torch.tensor(float(gt_box["x"]) + gt_box["width"], requires_grad=True)
	box1_y1 = torch.tensor(float(gt_box["y"]), requires_grad=True)
	box1_y2 = torch.tensor(float(gt_box["y"]) + gt_box["height"], requires_grad=True)

	box2_x1 = pred_box[0]
	box2_x2 = pred_box[2]
	box2_y1 = pred_box[1]
	box2_y2 = pred_box[3]

	# Determine midpoints of each box, then calculate Euclidean distance
	box1_x_mid = torch.div(torch.add(box1_x1, box1_x2), 2)
	box1_y_mid = torch.div(torch.add(box1_y1, box1_y2), 2)

	box2_x_mid = torch.div(torch.add(box2_x1, box2_x2), 2)
	box2_y_mid = torch.div(torch.add(box2_y1, box2_y2), 2)

	delta_x = torch.abs(torch.sub(box1_x_mid, box2_x_mid))
	delta_y = torch.abs(torch.sub(box1_y_mid, box2_y_mid))

	dist = torch.sqrt(torch.add(torch.square(delta_x), torch.square(delta_y)))

	return dist

def criterion(y_pred, y_data, lambda_iou, lambda_dist):
	batch_iou_list = None
	batch_dist_list = None
	#batch_diff_list = None

	y_hat_len_list = []
	y_gt_len_list = []

	for y_hat, y_gt in zip(y_pred, y_data): # iterating for each image

		# PRINT FOR DEBUGGING
		if y_hat is not None:
			y_hat_len_list.append(len(y_hat))#print("len(y_hat)", len(y_hat))
		else:
			y_hat_len_list.append(0)#print("len(y_hat)", 0)
		if y_gt is not None:
			y_gt_len_list.append(len(y_gt))#print("len(y_gt)", len(y_gt))
		else:
			y_gt_len_list.append(0)#print("len(y_hat)", 0)
		# PRINT FOR DEBUGGING

		# no bounding box for either y_pred or y_gt
		if y_hat is None or y_gt is None:
			continue # if either doesn't have any bounding box, skip the entry altogether

		# normal case where there is bounding box for both y_pred and y_gt
		else:
			iou_values = None
			dist_values = None

			for pred_box in y_hat: # iterating through each predicted boxes of this image 
				
				closest_prediction_iou = torch.tensor(-1.0, requires_grad=True) # cannot have negative IOU
				closest_prediction_dist = torch.tensor(-1.0, requires_grad=True) # cannot have negative IOU

				for gt_box in y_gt:
					current_iou = iou(gt_box, pred_box)
					current_dist = dist(gt_box, pred_box)

					if current_iou > closest_prediction_iou:
						closest_prediction_iou = current_iou

					if current_dist > closest_prediction_dist:
						closest_prediction_dist = current_dist

				# Append to the iou_values list tensor 
				closest_prediction_iou = torch.unsqueeze(closest_prediction_iou, 0)
				if iou_values is None:
					iou_values = closest_prediction_iou
				else:
					iou_values = torch.cat((iou_values, closest_prediction_iou))

				# Append to the dist_values list tensor
				closest_prediction_dist = torch.unsqueeze(closest_prediction_dist, 0)
				if dist_values is None:
					dist_values = closest_prediction_dist
				else:
					dist_values = torch.cat((dist_values, closest_prediction_dist))

			image_average_iou = torch.mean(iou_values)
			image_total_dist = torch.sum(dist_values)

			image_average_iou = torch.unsqueeze(image_average_iou, 0)
			if batch_iou_list is None:
				batch_iou_list = image_average_iou
			else:
				batch_iou_list = torch.cat((batch_iou_list, image_average_iou))

			image_total_dist = torch.unsqueeze(image_total_dist, 0)
			
			if batch_dist_list is None:
				batch_dist_list = image_total_dist
			else:
				batch_dist_list = torch.cat((batch_dist_list, image_total_dist))

	if batch_iou_list is None:
		batch_iou_list = torch.tensor(0.0, requires_grad=True)
	if batch_dist_list is None:
		batch_dist_list = torch.tensor(0.0, requires_grad=True)

	batch_avg_iou = torch.sub(torch.tensor(1), (torch.mean(batch_iou_list)))
	final_iou = torch.mul(batch_avg_iou, lambda_iou)

	batch_avg_dist = torch.mean(batch_dist_list)
	lambda_dist_pos = torch.abs(lambda_dist)
	final_dist = torch.mul(batch_avg_dist, lambda_dist_pos)

	loss = torch.add(final_iou, final_dist)

	return loss

def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('Running on device: {}'.format(device))

	if args.pretrained_mtcnn_model:
		mtcnn = MTCNN(
		    image_size=160, margin=0, min_face_size=20,
		    thresholds=[0.1, 0.2, 0.2], factor=0.709, post_process=True,
		    device=device, keep_all=True, pretrained_model_path=args.pretrained_mtcnn_model
		)
	else:
		mtcnn = MTCNN(
		    image_size=160, margin=0, min_face_size=20,
		    thresholds=[0.1, 0.2, 0.2], factor=0.709, post_process=True,
		    device=device, keep_all=True
		)

	train_image_list = os.listdir(args.meta_dir)
	train_dataset_size = len(train_image_list)

	# lambda variables
	lambda_iou = Variable(torch.tensor(1.0), requires_grad=True)
	lambda_dist = Variable(torch.tensor(0.0005), requires_grad=True)

	optimizer = torch.optim.SGD([{'params': mtcnn.parameters()}], lr=0.0001)

	# Training loop
	for epoch in range(args.num_epochs):

		# calculate how many iterations in the epoch
		iterations = int(train_dataset_size/BATCH_SIZE)

		# Train through every batch
		for iteration_num in range(iterations):
			x_data = []
			y_data = []

			batch_start_index = iteration_num*BATCH_SIZE

			for i in range(batch_start_index, batch_start_index+BATCH_SIZE):
				metadata_filename = train_image_list[i]
				metadata_file_pointer = open(os.path.join(args.meta_dir, metadata_filename), "r+")
				metadata_dict = json.load(metadata_file_pointer)
				metadata_file_pointer.close()
				label_id = metadata_dict["label_id"]
				
				image_filename = metadata_filename[:-5] + ".jpg"
				image = Image.open(os.path.join(args.images_dir, image_filename))
				x_data.append(image)

				label_filename = "{}.json".format(label_id)
				label_file_pointer = open(os.path.join(args.labels_dir, label_filename), "r+")
				label_dict = json.load(label_file_pointer)
				label_file_pointer.close()
				label_objects = label_dict["result"]["objects"]
				labels = []
				for label_object in label_objects:
					labels.append(label_object["shape"]["box"])
				y_data.append(labels)

			y_pred, probs = mtcnn.detect_face_tensor(x_data) # (batch_size, 4), (batch_size, 1)

			loss = criterion(y_pred, y_data, lambda_iou, lambda_dist)
			
			print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch, iteration_num, loss))

			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	torch.save(mtcnn.state_dict(), args.mtcnn_model_output_path)

# Driver code
if __name__ == "__main__":
	main()