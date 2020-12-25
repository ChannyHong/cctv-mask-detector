import os




image_list = os.listdir(".")
image_list.remove("count.py")
image_list.remove("rename.py")
image_list.remove(".DS_Store")

oxlist = [0] * 502

for image in image_list:
	dotindex = None
	for i, char in enumerate(image):
		if char == ".":
			dotindex = i
	num = image[:dotindex]
	intnum = int(num)
	oxlist[intnum] = 1

for i, ox in enumerate(oxlist):
	if ox == 0:
		print(i)

