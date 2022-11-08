from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.backend import argmax
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
import numpy as np
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

classifier = Sequential()

classifier.add(Convolution2D(32, 3,3, input_shape= (32,32,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 1152, activation = "relu"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 400, activation = "softmax"))

for layer in classifier.layers:
    print(layer.output_shape)


classifier.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ['accuracy'])

#Step2 

train_datagen = ImageDataGenerator (
	rescale = 1./255,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True)

test_datagen = ImageDataGenerator (rescale = 1./255)

training_set =train_datagen.flow_from_directory(
	'/Users/Toni/Desktop/EIT/KTH/RMSW/project/CNN/Dataset_not_occluded/training_set',
	target_size = (32,32),
	batch_size = 256,
	class_mode = 'categorical')

test_set=test_datagen.flow_from_directory(
	'/Users/Toni/Desktop/EIT/KTH/RMSW/project/CNN/Dataset_not_occluded/test_set',
	target_size = (32,32),
	batch_size = 256,
	class_mode = 'categorical')


for _ in range(5):
    img, label = training_set.next()
    print(argmax(label))
    print(img.shape) 
    plt.imshow(img[0])
    plt.show()


print(test_set.class_indices)

# Step 3 - training
#with increasing number of epochs, the accuracy will increase

classifier.fit(training_set, epochs=3)

#step 4 testing
print ("step 4")

test_image = image.load_img('/Users/Toni/Desktop/EIT/KTH/RMSW/project/CNN/Dataset_not_occluded/test_set/regulatory--maximum-speed-limit-15--g1/84058.jpg', target_size = (32,32))
test_image = image.img_to_array(test_image)
test_image = test_image/255.
test_image = np.expand_dims(test_image, axis = 0)

eval = classifier.evaluate(test_set)

print("test loss, test acc:", eval)

pred = classifier.predict(test_image)
print(argmax(pred[0]))