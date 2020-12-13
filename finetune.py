
import torch
from torch import nn, tensor
from torch.autograd import Variable

import json
import os
from mtcnn import MTCNN
from PIL import Image, ImageDraw

from torchviz import make_dot


BATCH_SIZE = 8


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
	# interArea = max(0, xB - xA) * max(0, yB - yA)
	interArea = torch.mul(torch.max(torch.tensor(0), torch.sub(xB, xA)), torch.max(torch.tensor(0), torch.sub(yB, yA)))

	# compute the area of both the prediction and ground-truth
	# rectangles
	#boxAArea = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
	boxAArea = torch.mul(torch.sub(box1_x2, box1_x1), torch.sub(box1_y2, box1_y1))

	#boxBArea = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
	boxBArea = torch.mul(torch.sub(box2_x2, box2_x1), torch.sub(box2_y2, box2_y1))

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	#iou = interArea / float(boxAArea + boxBArea - interArea)
	iou = torch.div(interArea, torch.sub(torch.add(boxAArea, boxBArea), interArea))
	# return the intersection over union value
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

			'''
			if y_hat is None and y_gt is None:
				continue # if no bounding box, just ignore the entry altogether
			elif y_hat is None:
				number_diff = torch.tensor([len(y_gt)]).type(torch.FloatTensor)
				if batch_diff_list is None:
					batch_diff_list = number_diff 
				else:
					batch_diff_list = torch.cat((batch_diff_list, number_diff))
			elif y_gt is None:
				number_diff = torch.tensor([len(y_hat)]).type(torch.FloatTensor)
				if batch_diff_list is None:
					batch_diff_list = number_diff 
				else:
					batch_diff_list = torch.cat((batch_diff_list, number_diff))
			'''

		# normal case where there is bounding box for both y_pred and y_gt
		else:
			iou_values = None #torch.tensor([[]], requires_grad=True)
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
			#image_average_dist = torch.mean(dist_values)
			image_total_dist = torch.sum(dist_values)

			#print("image_average_iou", image_average_iou)
			#print("image_total_dist", image_total_dist)

			#image_num_diff = torch.tensor([abs(len(y_hat) - len(y_gt))]).float()
			#image_num_diff.requires_grad = True


			image_average_iou = torch.unsqueeze(image_average_iou, 0)
			if batch_iou_list is None:
				batch_iou_list = image_average_iou
			else:
				batch_iou_list = torch.cat((batch_iou_list, image_average_iou))

			#image_average_dist = torch.unsqueeze(image_average_dist, 0)
			image_total_dist = torch.unsqueeze(image_total_dist, 0)
			
			if batch_dist_list is None:
				batch_dist_list = image_total_dist
			else:
				batch_dist_list = torch.cat((batch_dist_list, image_total_dist))

			'''
			if batch_dist_list is None:
				batch_dist_list = image_average_dist
			else:
				batch_dist_list = torch.cat((batch_dist_list, image_average_dist))

			if batch_diff_list is None:
				batch_diff_list = image_num_diff 
			else:
				batch_diff_list = torch.cat((batch_diff_list, image_num_diff))
			'''

	if batch_iou_list is None:
		batch_iou_list = torch.tensor(0.0, requires_grad=True)
	if batch_dist_list is None:
		batch_dist_list = torch.tensor(0.0, requires_grad=True)


	#batch_avg_iou = torch.neg(torch.mean(batch_iou_list))
	#batch_avg_iou = torch.div(torch.tensor(1), (torch.mean(batch_iou_list)))
	batch_avg_iou = torch.sub(torch.tensor(1), (torch.mean(batch_iou_list)))
	final_iou = torch.mul(batch_avg_iou, lambda_iou)

	batch_avg_dist = torch.mean(batch_dist_list)
	lambda_dist_pos = torch.abs(lambda_dist)
	final_dist = torch.mul(batch_avg_dist, lambda_dist_pos)

	print("lambda_iou", lambda_iou)
	print("lambda_dist", lambda_dist)

	print("average_y_hat_len", float(sum(y_hat_len_list)) / float(len(y_hat_len_list)))
	print("average_y_gt_len", float(sum(y_gt_len_list)) / float(len(y_gt_len_list)))

	print("final_iou", final_iou)
	print("final_dist", final_dist)

	#batch_avg_diff = torch.mean(batch_diff_list)
	#lambda_diff = Variable(torch.tensor(1.0), requires_grad=True)
	#final_diff = torch.mul(batch_avg_diff, lambda_diff)

	
	#loss is avg_iou + avg_diff + avg_dist
	loss = torch.add(final_iou, final_dist)
	#loss = torch.add(temp, final_diff)




	return loss
	# identify how 'off' the predictions were on the following factors
	# 1. area of the unaccounted box, compared to the average area of size of the ground truths
	# 2. Euclidean distance of the center point of the box to the closest ground truth box 






def main():
	#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	device = "cpu"
	print('Running on device: {}'.format(device))

	mtcnn = MTCNN(
	    image_size=160, margin=0, min_face_size=20,
	    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
	    device=device, keep_all=True#, pretrained_model_path="test.pt"
	)

	train_image_list = os.listdir("../labeled_data/image_metadata")
	train_dataset_size = len(train_image_list)

	#train_image_list.sort()

	# lambda variables
	lambda_iou = Variable(torch.tensor(1.0), requires_grad=True)
	lambda_dist = Variable(torch.tensor(0.0), requires_grad=True)

	#criterion = torch.nn.MSELoss(reduction='sum')
	#params = list(mtcnn.parameters()) + [lambda_iou, lambda_dist]
	#optimizer = torch.optim.SGD(params, lr=0.0001)

	optimizer = torch.optim.SGD([
		{'params': mtcnn.parameters()}#,
		#{'params': [lambda_iou, lambda_dist], 'lr': 0.00001}
		], lr=0.0001)

	# Training loop
	for epoch in range(20):

		# calculate how many iterations in the epoch
		iterations = int(train_dataset_size/BATCH_SIZE)

		# Train through every batch
		for iteration_num in range(iterations):
			x_data = []
			y_data = []

			batch_start_index = iteration_num*BATCH_SIZE

			for i in range(batch_start_index, batch_start_index+BATCH_SIZE):
				metadata_filename = train_image_list[i]
				metadata_file_pointer = open(os.path.join("../labeled_data/image_metadata", metadata_filename), "r+")
				metadata_dict = json.load(metadata_file_pointer)
				metadata_file_pointer.close()
				label_id = metadata_dict["label_id"]
				
				image_filename = metadata_filename[:-5]
				image = Image.open(os.path.join("../labeled_data/images", image_filename))
				x_data.append(image)

				label_filename = "{}.json".format(label_id)
				label_file_pointer = open(os.path.join("../labeled_data/labels", label_filename), "r+")
				label_dict = json.load(label_file_pointer)
				label_file_pointer.close()
				label_objects = label_dict["result"]["objects"]
				labels = []
				for label_object in label_objects:
					labels.append(label_object["shape"]["box"])
				y_data.append(labels)

			y_pred, probs = mtcnn.detect_face_tensor(x_data) # (batch_size, 4), (batch_size, 1)

			loss = criterion(y_pred, y_data, lambda_iou, lambda_dist)
			
			#model_arch = make_dot(loss)
			#model_arch.format = 'png'
			#model_arch.render("loss_graphs/epoch{}-iter{}-loss.png".format(epoch, iteration_num))

			print('Epoch: {} | Iteration: {} | Loss: {} \n.\n.\n.\n.\n.'.format(epoch, iteration_num, loss))

			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if loss < 0.01:
				torch.save(mtcnn.state_dict(), "./test_ep{}_iter{}_loss{}.pt".format(epoch, iteration_num, loss))

	torch.save(mtcnn.state_dict(), "./test_end.pt")


if __name__ == "__main__":
	main()
