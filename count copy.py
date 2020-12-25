import os




image_list = os.listdir(".")
image_list.remove("count.py")
image_list.remove(".DS_Store")

oxlist = [0] * 601

for image in image_list:
	num = image[:-5]
	intnum = int(num)
	oxlist[intnum] = 1

for i, ox in enumerate(oxlist):
	if ox == 0:
		print(i)

