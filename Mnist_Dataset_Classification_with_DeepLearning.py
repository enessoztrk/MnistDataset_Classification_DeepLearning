# Mnist Dataset Classification with Deep Learning

# Library
import random
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import load_model


# Train-Test Split
(train_images, train_labels) = mnist.load_data()[0]
(test_images, test_labels) = mnist.load_data()[1]

print("Shape of Train Images", train_images.shape)
print("Shape of Test Images", test_images.shape)
# Shape of Train Images (60000, 28, 28)
# Shape of Test Images (10000, 28, 28)


# Displaying Data
plt.imshow(train_images[20], cmap="gray_r")
plt.show()


# Preparing Data
"""
1. 3D --> 4D Numpy Array
2. Normalization: /255
"""

# 3D --> 4D Numpy Array
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
print("Shape of Train Images", train_images.shape)
print("Shape of Test Images", test_images.shape)

# Normalization: /255
train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

train_images /= 255
test_images /= 255
# Shape of Train Images (60000, 28, 28, 1)
# Shape of Test Images (10000, 28, 28, 1)

input_shape = (28,28,1)


# Creating Network
model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))


# Compiling Model
model.compile(optimizer="adam",
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"])

model.summary()
# conv2d (Conv2D)               (None, 26, 26, 28)
# max_pooling2d (MaxPooling2D)  (None, 13, 13, 28)
# flatten (Flatten)             (None, 4732)
# dense (Dense)                 (None, 128)
# dropout (Dropout)             (None, 128)
# dense_1 (Dense)               (None, 10)


# Fitting Model
history = model.fit(x = train_images,
                    y = train_labels,
                    epochs = 10)


# Evaluating Model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0672 - accuracy: 0.9851
# Test Loss:  0.06720476597547531
# Test Accuracy:  0.9850999712944031

history_dict = history.history
print("Keys: ", history_dict.keys())
# Keys:  dict_keys(['loss', 'accuracy'])


# loss, accuracy graph
epochs = range(1,11)
loss = history_dict["loss"]
accuracy = history_dict["accuracy"]

plt.plot(epochs, loss)
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(epochs, accuracy)
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# Saving Model
model.save("mnist_model.h5")


# Loading Model and Prediction
model = load_model("mnist_model.h5")
i = random.randint(1,5000)
prediction = model.predict(test_images[i].reshape(1,28,28,1))
print("Predicted Number: ",prediction.argmax())
plt.imshow(test_images[i].reshape(28,28), cmap="gray_r")
plt.show()
# Predicted Number:  6
# images6...Good!