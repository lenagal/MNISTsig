import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
import os

learning_rate = 0.01
epochs=50

#signature used up to level
level=3

#calculate signature data size
fractal_depth=2
sample_size=3
signature_dim=(pow(2,level+1)-1)
#number of paths in the dyadic tree
#if training only with lowest level of dyadic samples set
sampled_pieces=2**fractal_depth
#sampled_pieces=(pow(2,fractal_depth+1)-1)


#read data
parent_dir='/Users/user/Google Drive/atom programming/mnistseq/'
data_dir=os.path.join(parent_dir,'sampled_leaves/')

train_images=np.load(os.path.join(data_dir,'train_images.npy'))
train_labels=np.load(os.path.join(data_dir,'train_labels.npy'))
test_images=np.load(os.path.join(data_dir,'test_images.npy'))
test_labels=np.load(os.path.join(data_dir,'test_labels.npy'))

print(train_images.shape)

train_images=train_images.reshape(60000,sampled_pieces*signature_dim)
test_images=test_images.reshape(10000,sampled_pieces*signature_dim)
print(test_images.shape,train_images.shape)

#Reserve 10000 samples for validation
train_images_val=train_images[-1000:]
train_labels_val=train_labels[-1000:]
train_images=train_images[:-1000]
train_labels=train_labels[:-1000]



#define and train model
model = Sequential([
    Dense(128, activation='relu',input_shape=(sampled_pieces*signature_dim,)),
    Dense(10, activation='softmax')
])

model.compile(
  optimizer=Adam(lr=learning_rate),
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

print("Fit on training data")
history=model.fit(
  train_images,
  train_labels,
  batch_size=128,
  epochs=epochs,
  verbose=2,
  validation_data=(train_images_val,train_labels_val)
)

print("evaluate on test data")
results=model.evaluate(test_images,test_labels,batch_size=128)
print("test loss, test acc:",results)

#save model
save_dir = '/Users/user/Google Drive/atom programming/mnistseq/models'
model_name = 'mnist_seq_window'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# Predict on test images.
predictions = model.predict(test_images)

# Analyse and print model's predictions.
errormatrix=np.zeros([10,10])
incidence_count=np.zeros([10])
error_count=np.zeros([10])
for i in range(0,test_images.shape[0]):
    incidence_count[np.argmax(test_labels[i])]+=1
    if np.argmax(predictions[i])!=np.argmax(test_labels[i]):
        error_count[np.argmax(test_labels[i])]+=1
        errormatrix[np.argmax(test_labels[i])][np.argmax(predictions[i])]+=1

print("Confusion Matrix\n",errormatrix)

print("Overall incidence by digit\n",incidence_count)

print("Errors by digit \n", error_count)

#print("",errormatrix-errormatrix.T)
