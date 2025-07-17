import numpy as np
import os
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import keras

plants = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Tomato_Bacterial_spot',
          'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
          'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

X = []
Y = []
'''
for i in range(len(plants)):
    for root, dirs, directory in os.walk('PlantVillage/'+plants[i]):
        for j in range(len(directory)):
            img = cv2.imread('PlantVillage/'+plants[i]+"/"+directory[j])
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            X.append(im2arr)
            Y.append(i)
            print('PlantVillage/'+plants[i]+"/"+directory[j])
                    
np.save("model/myimg_data.txt",X)
np.save("model/myimg_label.txt",Y)


X = np.load("model/myimg_data.txt.npy")
Y = np.load("model/myimg_label.txt.npy")

X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape)
Y = to_categorical(Y)
print(Y.shape)
img = X[20].reshape(64,64,3)
cv2.imshow('ff',cv2.resize(img,(250,250)))
cv2.waitKey(0)
print("shape == "+str(X.shape))
print("shape == "+str(Y.shape))
print(Y)
X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

classifier = Sequential() #alexnet transfer learning code here
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 15, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(classifier.summary())
classifier.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
classifier.save_weights('model/model_weights.h5')            
model_json = classifier.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)

'''

with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model/model_weights.h5")
loaded_model._make_predict_function()   
print(loaded_model.summary())



img = cv2.imread('download.jpg')
img = cv2.resize(img, (64,64))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,64,64,3)
X = np.asarray(im2arr)
X = X.astype('float32')
X = X/255
preds = loaded_model.predict(X)
print(str(preds)+" "+str(np.argmax(preds)))
predict = np.argmax(preds)
print(plants[predict])
img = im2arr.reshape(64,64,3)
cv2.imshow(plants[predict],cv2.resize(img,(250,250)))
cv2.waitKey(0)





