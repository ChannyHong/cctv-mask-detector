from PIL import Image
import os


# Open Paddington


images = os.listdir('../kaggle-dataset/images/original/')

for image in images:

	img = Image.open("../kaggle-dataset/images/original/{}".format(image))

	# Resize smoothly down to 16x16 pixels
	imgNew = img.resize((1280,720),resample=Image.BILINEAR) # pixelated
	#imgNew = img.resize((640,360),resample=Image.BILINEAR) # pixelated-x2
	#imgNew = img.resize((320,180),resample=Image.BILINEAR) # pixelated-x4


	# Scale back up using NEAREST to original size
	result = imgNew.resize(img.size,Image.NEAREST)

	# Save
	result.save('../kaggle-dataset/images/pixelated/{}'.format(image))




'''

from PIL import Image

# Open Paddington
img = Image.open("../kaggle-dataset/images_test/all/0870.jpg")

# Resize smoothly down to 16x16 pixels
#imgNew = img.resize((1280,720),resample=Image.BILINEAR) # pixelated
#imgNew = img.resize((640,360),resample=Image.BILINEAR) # pixelated-x2
imgNew = img.resize((320,180),resample=Image.BILINEAR) # pixelated-x4


# Scale back up using NEAREST to original size
result = imgNew.resize(img.size,Image.NEAREST)

# Save
result.save('../kaggle-dataset/images_test/all/pixelated.jpg')

'''