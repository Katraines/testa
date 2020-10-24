#importing os 
import os

#path of the folder
path = ('faces')

#using listdir() method to list the files of the folder
faces = os.listdir(path)

#taking a loop to remove all the images
#using ".jpg" extension to remove only png images
#using os.remove() method to remove the files
for images in faces:
    if images.endswith(".jpg"):
        os.remove(os.path.join(path, images))

print("Ta-da! I cleaned all your files. It's very unlikely, but I listed any remaining files below:")
print(os.listdir('faces'))