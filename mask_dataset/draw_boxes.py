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
    device=device, keep_all=True, pretrained_model_path="mtcnn_finetuned.pt"
)

train_image_list = os.listdir("test_dataset")
train_image_list.remove(".DS_Store")


train_dataset_size = len(train_image_list)
#train_image_list.sort()

#image_list.sort()




for i in range(train_dataset_size):
	image_filename = train_image_list[i]

	image = Image.open(os.path.join("test_dataset", image_filename))
				

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

	image_boxed.save(os.path.join("test_dataset_boxes", image_filename))




'''
x_data.reverse()

boxes, probs = mtcnn.detect(x_data)

print("Boxes: ", boxes)
print("Probs: ", probs)

'''








