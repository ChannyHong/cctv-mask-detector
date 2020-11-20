import face_recognition


image = face_recognition.load_image_file("kaggle-dataset/images_test/all/36-frame220.jpg")


face_locations = face_recognition.face_locations(image, model="cnn") # 'cnn' model is much better than not using it
print(face_locations)