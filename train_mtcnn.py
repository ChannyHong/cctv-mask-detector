import os
import torch

from facenet_pytorch import MTCNN

from PIL import Image, ImageDraw


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device, keep_all=True
)



data_dir = 'test_dataset'
image_files = os.listdir(data_dir)

image_files.remove('.DS_Store')


for image_file in image_files:


    image = Image.open(os.path.join(data_dir, image_file))

    boxes, _ = mtcnn.detect(image)

    image_boxed = image.copy()
    draw = ImageDraw.Draw(image_boxed)

    if not boxes is None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)

    image_boxed.save(os.path.join("boxed_images", image_file))

