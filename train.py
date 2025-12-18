import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

IMG_SIZE=(224,224)
BATCH=32

train_path='YOUR_TRAIN_PATH'
test_path='YOUR_TEST_PATH'

train_gen=ImageDataGenerator(rescale=1/255).flow_from_directory(
    train_path,target_size=IMG_SIZE,batch_size=BATCH,class_mode='binary')

test_gen=ImageDataGenerator(rescale=1/255).flow_from_directory(
    test_path,target_size=IMG_SIZE,batch_size=BATCH,class_mode='binary')

base=EfficientNetB0(weights='imagenet',include_top=False,input_shape=(224,224,3))
base.trainable=False

model=models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_gen,validation_data=test_gen,epochs=10)
model.save('crack_model.h5')
