import torch
import pandas as pd

from facenet_pytorch import MTCNN, InceptionResnetV1


resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



DIR_INPUT = "../kaggle-dataset/"
DIR_IMAGES = DIR_INPUT + "Medical mask/Medical mask/Medical Mask/images/"



df = pd.read_csv(DIR_INPUT + "train.csv")

print(df.head())



unq_values = df["name"].unique()
print("Total Records: ", len(df))
print("Unique Images: ",len(unq_values))

null_values = df.isnull().sum(axis = 0)
print("\n> Null Values in each column <")
print(null_values)