from PIL import Image

# Open Paddington
img = Image.open("../kaggle-dataset/images_test/all/36-frame220.jpg")

# Resize smoothly down to 16x16 pixels
imgNew = img.resize((1280,720),resample=Image.BILINEAR)

# Scale back up using NEAREST to original size
result = imgNew.resize(img.size,Image.NEAREST)

# Save
result.save('result.png')