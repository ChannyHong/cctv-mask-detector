import os


image_list = os.listdir('.')
#image_list.remove('count.py')
image_list.remove('rename.py')
#image_list.remove('.DS_Store')

counter = 1


print(image_list)


for image in image_list:
	os.rename(image, str(counter) + ".jpeg")
	counter += 1


