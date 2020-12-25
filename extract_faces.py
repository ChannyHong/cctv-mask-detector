import torch
import os
#from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import json

from mtcnn import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.05, 0.1, 0.1], factor=0.709, post_process=True,
    device=device, keep_all=True, pretrained_model_path=None
)

train_image_list = os.listdir("../labeled_data/image_metadata")
train_dataset_size = len(train_image_list)

train_image_list.sort()

#train_image_list.remove(".DS_Store")
#image_list.sort()




for i in range(train_dataset_size):
	metadata_filename = train_image_list[i]
	metadata_file_pointer = open(os.path.join("../labeled_data/image_metadata", metadata_filename), "r+")
	metadata_dict = json.load(metadata_file_pointer)
	metadata_file_pointer.close()
	label_id = metadata_dict["label_id"]


	image_filename = metadata_filename[:-5]
	image = Image.open(os.path.join("../labeled_data/images", image_filename))
				


	image_boxed = image.copy()
	draw = ImageDraw.Draw(image_boxed)

	#x_data.append(image)

	#output = mtcnn.detect_face_tensor(image)
	boxes, _ = mtcnn.detect(image)

	#print("boxes is tensor: ", torch.is_tensor(boxes))
	#print("probs is tensor: ", torch.is_tensor(probs))

	#print(image_file)
	#print(output)


	if not boxes is None:
		for box in boxes:
			draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)

	image_boxed.save(os.path.join("../labeled_data/draw_boxes", image_filename))




'''
x_data.reverse()

boxes, probs = mtcnn.detect(x_data)

print("Boxes: ", boxes)
print("Probs: ", probs)

'''








