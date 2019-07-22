
# Imports
import os
#; os.environ['KERAS_BACKEND']='theano'
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

# Loading the data
raw_data = load_files(os.getcwd() + r'/UCF11_updated_mpg', shuffle=False)



files = raw_data['filenames']


targets = raw_data['target']


# Randomly dividing the whole data into training (66.67%) and testing (33.33%) data
train_files, test_files, train_targets, test_targets = train_test_split(files, targets, test_size=1/4, random_state=0)




# Generic details about the data
print('Total number of videos:', len(files))
print('\nNumber of videos in training data:', train_files.shape[0])
#print('Number of videos in validation data:', valid_files.shape[0])
print('Number of videos in test data:', test_files.shape[0])


# ### Description of the class labels

# In[2]:


print('The categorical labels are converted into integers.\nFollowing is the mapping - \n')
for label in zip(range(11), raw_data['target_names']):
	print(label)



# Displaying the first 5 videos (paths) in the training data along with their labels
# (path of video, class label)
for pair in zip(train_files[:5], train_targets[:5]):
	print(pair)



# Imports
import numpy as np
import matplotlib.pyplot as plt
from utils import Videos
#get_ipython().run_line_magic('matplotlib', 'inline')


# The path of a sample video in the training data
sample_files = train_files[3:4]


# An object of the class 'Videos'
reader = Videos(target_size=None,
				to_gray=False)




# Imports
import numpy as np
from keras.utils import to_categorical
from utils import Videos

# An object of the class `Videos` to load the data in the required format
reader = Videos(target_size=(128, 128),
				to_gray=True,
				max_frames=3,
				extract_frames='first',
				required_fps=1,
				normalize_pixels=(-1, 1))



#Reading training videos and one-hot encoding the training labels
X_train = reader.read_videos(train_files)
y_train = to_categorical(train_targets, num_classes=11)
print('Shape of training data:', X_train.shape)
print('Shape of training labels:', y_train.shape)


# In[ ]:


# Reading validation videos and one-hot encoding the validation labels
# X_valid = reader.read_videos(valid_files)
# y_valid = to_categorical(valid_targets, num_classes=6)
# print('Shape of validation data:', X_valid.shape)
# print('Shape of validation labels:', y_valid.shape)


# In[ ]:


# Reading testing videos and one-hot encoding the testing labels
X_test = reader.read_videos(test_files)
y_test = to_categorical(test_targets, num_classes=11)
print('Shape of testing data:', X_test.shape)
print('Shape of testing labels:', y_test.shape)


# ### Model - 3
#
# This model has one more layer than the previous models, but it produces a deeper vector, thus better at enconding the content of the video than the previous models.
#
# The dense layers at the end are not changed and are same as that of *Model-2*.

# In[ ]:


# Imports
#from keras.models import Sequential
from numpy import array 
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, BatchNormalization,GlobalMaxPooling3D, TimeDistributed, Bidirectional
from keras.layers import Dense, Dropout, LeakyReLU, Flatten, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.convolutional_recurrent import ConvLSTM2D
#from keras.optimizers import Nadam

# Using the Sequential Model
model = Sequential()

# model.add(ConvLSTM2D(filters=256, kernel_size=(3, 3),
# 				   input_shape=(None, 35, 128, 128),
# 				   padding='same', return_sequences=True))
# model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))




# Adding Alternate convolutional and pooling layers
model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
				 input_shape=X_train.shape[1:]))
#model.add(LeakyReLU())

model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))

model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
#model.add(LeakyReLU())


model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))

model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
#model.add(LeakyReLU())


model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2), padding='same'))

model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))


model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',activation='relu'))
#model.add(LeakyReLU())


model.add(MaxPooling3D(pool_size=2, strides=(2,2,2), padding='same'))

model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))

model.add(Conv3D(filters=1024, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
#model.add(LeakyReLU())

model.add(MaxPooling3D(pool_size=2, strides=(2,2,2), padding='same'))

model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))


model.add(Conv3D(filters=1024, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
#model.add(LeakyReLU())

model.add(MaxPooling3D(pool_size=2, strides=(2,2,2), padding='same'))

model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))

# model.add(Conv3D(filters=4096, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
# #model.add(LeakyReLU(alpha= 0.3))

# model.add(MaxPooling3D(pool_size=2, strides=(2,2,2), padding='same'))

# model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))


# model.add(Conv3D(filters=4096, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
# # model.add(LeakyReLU(alpha= 0.3))

# model.add(MaxPooling3D(pool_size=2, strides=(2,2,2), padding='same'))

# model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))

# model.add(TimeDistributed(Conv3D(filters=4096, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='same',activation='relu')))
# #model.add(LeakyReLU(alpha= 0.3))

# model.add(TimeDistributed(MaxPooling3D(pool_size=2, strides=(2,2,2), padding='same')))

# model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))

# model.add(ConvLSTM2D(filters=256, kernel_size=(3, 3),
# 				   input_shape=(None, 1, 2, 2),
#  				   padding='same', return_sequences=True))
# model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))

# #model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3),
# #					 padding='same', return_sequences=True))
# model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))

model.add(Flatten())


# model.add(Conv3D(filters=4096, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding='same',activation='relu'))
# #model.add(LeakyReLU(alpha= 0.3))

# model.add(MaxPooling3D(pool_size=2, strides=(2,2,2), padding='same'))

# model.add(BatchNormalization(epsilon= 1e-06, axis=3, mode=0, momentum=0.9, weights=None))


# A global average pooling layer to get a 1-d vector
# The vector will have a depth (same as number of elements in the vector) of 1024
#model.add(GlobalAveragePooling3D())



#model.add(GlobalMaxPooling3D())



#LSTM



# Hidden layer
model.add(Dense(units=256, activation='relu'))
#model.add(LeakyReLU(alpha= 0.3))
#model.add(Dropout(0.5))
#extra
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu')) #kernel_regularizer=regularizers.l2(0.01)

#model.add(LeakyReLU(alpha= 0.3))

# Dropout Layer
#model.add(Dropout(0.3))

# Output layer
model.add(Dense(units=11, activation='softmax'))

model.summary()



# Imports
from keras.callbacks import ModelCheckpoint

#optimizer = Nadam(lr=0.01)

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

# Saving the model that performed the best on the validation set
checkpoint = ModelCheckpoint(filepath='Model_ucf11.weights.best.hdf5', monitor='val_loss', save_best_only=True,
	                          save_weights_only=False,mode='auto', period=1, verbose=1)

# Training the model for 40 epochs
history = model.fit(X_train, y_train, batch_size=16, epochs=60,validation_data=(X_test, y_test), verbose=2, callbacks=[checkpoint])  


# Loading the model that performed the best on the validation set
#model.load_model('Model_3.weights.best.hdf5')
model.load_weights('Model_ucf11.weights.best.hdf5')

# Testing the model on the Test data
(loss, accuracy) = model.evaluate(X_test, y_test, batch_size=16, verbose=0)

print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))


# ## Model - 3 Performance
#
# The model gave an **Accuracy of 64.5% on the test data**.
#
# This time, the model gave a higher accuracy than the previous models, despite using 5 times lesser data for training.
#
# ### Learning Curve


# Making the plot larger
# plt.figure(figsize=(12, 8))

# loss = history.history['loss']                          # Loss on the training data
# val_loss = history.history['val_loss']                  # Loss on the validation data
# epochs = range(1, 41)

# plt.plot(epochs, loss, 'ro-', label='Training Loss')
# plt.plot(epochs, val_loss, 'go-', label = 'Validation Loss')
# plt.legend()



