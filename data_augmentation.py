BATCH_SIZE=32
INPUT_IMAGE_ROWS=160
INPUT_IMAGE_COLS=320
INPUT_IMAGE_CHANNELS=3
AUGMENTATION_NUM_BINS=200
NUM_EPOCHS=5
AUGMENTATION_BIN_MAX_PERC=7
AUGMENTATION_FACTOR=3



import csv
import cv2
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import Callback
import threading
import math
from keras.preprocessing.image import *
import matplotlib.pyplot as plt

print("\nLoading the dataset from file ...")

def load_dataset(file_path):
    dataset = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                dataset.append({'center':line[0], 'left':line[1], 'right':line[2], 'steering':float(line[3]),
                            'throttle':float(line[4]), 'brake':float(line[5]), 'speed':float(line[6])})
            except:
                print("ok")
                continue # some images throw error during loading
    return dataset

dataset = load_dataset('C:\\Users\\kiit1\\Documents\\dataset\\data\\data\\driving_log.csv')
print("Loaded {} samples from file {}".format(len(dataset),'C:\\Users\\kiit1\\Documents\\dataset\\data\\data\\driving_log.csv'))



print("Partioning the dataset:")

shuffle(dataset)

#partitioning data into 80% training, 19% validation and 1% testing

X_train,X_validation=train_test_split(dataset,test_size=0.2)
X_validation,X_test=train_test_split(X_validation,test_size=0.05)

test_data=open("test.csv","w")
#write.csv(X_train,"test_data.csv")
feildnames=['center','left','right','steering','throttle','brake','speed']
writer=csv.DictWriter(test_data,fieldnames=feildnames)
writer.writeheader()
for row in X_test:
 #writer.writerow({'center':row[0],'left':row[1],'right':row[2],'steering':row[3],'throttle':row[4],'break':row[5],'speed':row[5]})
 writer.writerow(row)


print("X_train has {} elements.".format(len(X_train)))
print("X_validation has {} elements.".format(len(X_validation)))
print("X_test has {} elements.".format(len(X_test)))
print("Partitioning the dataset complete.")


from PIL import Image

#@threadsafe_generator
def generate_batch_data(dataset, batch_size = 16):
    #print("in generate_batch_data")
    global augmented_steering_angles
    global epoch_steering_count
    global epoch_bin_hits
    batch_images = np.zeros((batch_size, INPUT_IMAGE_ROWS, INPUT_IMAGE_COLS, INPUT_IMAGE_CHANNELS))
    batch_steering_angles = np.zeros(batch_size)
    augmented_steering=[]


    while 1:
        for batch_index in range(batch_size):

            # select a random image from the dataset
            image_index = np.random.randint(len(dataset))
            #print("Image index=")
            #print(image_index)
            image_data = dataset[image_index]
            #print(image_data)
            #image, steering_angle = load_and_augment_image(image_data)


            #image_data.show()



            while 1:

                #image, steering_angle = load_and_augment_image(image_data)
                try:
                    image, steering_angle = load_and_augment_image(image_data)
                    augmented_steering.append(float(steering_angle))
                except:
                    continue


                bin_idx = int (steering_angle * AUGMENTATION_NUM_BINS / 2)

                if( epoch_bin_hits[bin_idx] < epoch_steering_count/AUGMENTATION_NUM_BINS*AUGMENTATION_BIN_MAX_PERC
                    or epoch_steering_count<500 ):

                    #print("Here")
                    batch_images[batch_index] = image
                    batch_steering_angles[batch_index] = steering_angle
                    augmented_steering_angles.append(steering_angle)

                    epoch_bin_hits[bin_idx] = epoch_bin_hits[bin_idx] + 1
                    epoch_steering_count = epoch_steering_count + 1
                    break
                else:
                    print("Not taking")

        yield batch_images, batch_steering_angles








print("\nTraining the model ...")

class LifecycleCallback(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global epoch_steering_count
        global epoch_bin_hits
        global bin_range
        epoch_steering_count = 0
        epoch_bin_hits = {k:0 for k in range(-bin_range, bin_range)}

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        print('Beginning training')
        self.losses = []

    def on_train_end(self, logs={}):
        print('Ending training')

# Compute the correct number of samples per epoch based on batch size
def compute_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil(num_batches)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch







def load_and_augment_image(image_data, side_camera_offset=0.2):
    #print("in load_and_augment_image")
    #co=1
    # select a value between 0 and 2 to swith between center, left and right image
    index = np.random.randint(3)
    if (index==0):
        image_file = image_data['left'].strip()
        angle_offset = side_camera_offset
    elif (index==1):
        image_file = image_data['center'].strip()
        angle_offset = 0.
    elif (index==2):
        image_file = image_data['right'].strip()
        angle_offset = - side_camera_offset

    steering_angle = image_data['steering'] + angle_offset

    image = cv2.imread(image_file)
    #image = cv2.imread("C://Users//kiit1//Downloads//data//dataset//IMG/center_2016_12_01_13_46_36_366.jpg")
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply a misture of several augumentation methods
    image, steering_angle = random_transform(image, steering_angle)

    return image, steering_angle


augmented_steering_angles = []

epoch_steering_count = 0
bin_range = int(AUGMENTATION_NUM_BINS / 4 * 3)
epoch_bin_hits = {k:0 for k in range(-bin_range, bin_range)}




"""
******************************************************************************************************************
                  ***********************************************************************

                                DATA PREPROCESSING AND AUGMENTATION

                 *************************************************************************
********************************************************************************************************************

"""





#flips image about y-axis
def horizontal_flip(image,steering_angle):
   # print("in horizontal_flip")
    flipped_image=cv2.flip(image,1);
    steering_angle=-steering_angle
    return flipped_image,steering_angle

def translate(image,steering_angle,width_shift_range=50.0,height_shift_range=5.0):
    #print("in translate")
    tx = width_shift_range * np.random.uniform() - width_shift_range / 2
    ty = height_shift_range * np.random.uniform() - height_shift_range / 2

     # new steering angle
    steering_angle += tx / width_shift_range * 2 * 0.2

    transformed_matrix=np.float32([[1,0,tx],[0,1,ty]])
    rows,cols=(image.shape[0],image.shape[1])

    translated_image=cv2.warpAffine(image,transformed_matrix,(cols,rows))
    return translated_image,steering_angle

def brightness(image,bright_increase=None):
    #print("in brightness")
    if(image.shape[2]>1):
        image_hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    else:
        image_hsv=image

    if bright_increase:
        image_hsv[:,:,2] += bright_increase
    else:
        bright_increase = int(30 * np.random.uniform(-0.3,1))
        image_hsv[:,:,2] = image[:,:,2] + bright_increase

    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image

def rotation(image,rotation_range=5):
    #print("in rotation")
    image=random_rotation(image,rotation_range);
    return image


# Shift range for each channels
def channel_shift(image, intensity=30, channel_axis=2):
    #print("in channel_shift")
    image = random_channel_shift(image, intensity, channel_axis)
    return image

# Crop and resize the image
def crop_resize_image(image, cols=INPUT_IMAGE_COLS, rows=INPUT_IMAGE_ROWS, top_crop_perc=0.1, bottom_crop_perc=0.2):
    #print("in crop_resize_image")
    height = image.shape[0]
    width= image.shape[1]

    # crop top and bottom
    top_rows = int(height*top_crop_perc)
    bottom_rows = int(height*bottom_crop_perc)
    image = image[top_rows:height-bottom_rows, 0:width]

    # resize to the final sizes even the aspect ratio is destroyed
    image = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_LINEAR)
    return image


# Apply a sequence of random tranformations for a better generalization and to prevent overfitting
def random_transform(image, steering_angle):
   # print("in random_transform")
    # all further transformations are done on the smaller image to reduce the processing time
    image = crop_resize_image(image)

    # every second image is flipped horizontally
    if np.random.random() < 0.5:
        image, steering_angle = horizontal_flip(image, steering_angle)

    image, steering_angle = translate(image, steering_angle)
    image = rotation(image)
    image = brightness(image)
    image = channel_shift(image)

    return img_to_array(image), steering_angle



"""
******************************************************************************************************************
                  ***********************************************************************

                                CONVOLUTIONAL NUERAL NETWORK

                 *************************************************************************
********************************************************************************************************************

"""

from keras.models import Sequential, Model
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.advanced_activations import ELU
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam


print("\nBuilding and compiling the model ...")

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(INPUT_IMAGE_ROWS, INPUT_IMAGE_COLS, INPUT_IMAGE_CHANNELS)))
    # Conv Layer1 of 16 filters having size(8, 8) with strides (4,4)
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
    # Conv Layer1 of 32 filters having size(5, 5) with strides (2,2)
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
    # Conv Layer1 of 64 filters having size(5, 5) with strides (2,2)
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(512,activation='elu'))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.summary()
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)

#saving the model
model_json=model.to_json()
with open("model.json","w")as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")



lifecycle_callback = LifecycleCallback()

train_generator = generate_batch_data(X_train, BATCH_SIZE)
validation_generator = generate_batch_data(X_validation, BATCH_SIZE)

samples_per_epoch = compute_samples_per_epoch((len(X_train)*AUGMENTATION_FACTOR), BATCH_SIZE)
nb_val_samples = compute_samples_per_epoch((len(X_validation)*AUGMENTATION_FACTOR), BATCH_SIZE)

history = model.fit_generator(train_generator,
                              validation_data = validation_generator,
                              samples_per_epoch = ((len(X_train) // BATCH_SIZE ) * BATCH_SIZE) * 2,
                              nb_val_samples = ((len(X_validation) // BATCH_SIZE ) * BATCH_SIZE) * 2,
                              nb_epoch = NUM_EPOCHS, verbose=1,
                              )


# list all data in history
print(history.history.keys())

# summarize history for epoch loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.show()

# summarize history for batch loss
batch_history = lifecycle_callback.losses
plt.plot(batch_history)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('batches')
plt.show()

print("\nTraining the model ended.")





"""
******************************************************************************************************************
                  ***********************************************************************

                                                TRAINING

                 *************************************************************************
********************************************************************************************************************

"""

#plotting augmented train data

def plot_steering_histogram(steerings, title, num_bins=100):
    plt.hist(steerings, num_bins)
    plt.title(title)
    plt.xlabel('Steering Angles')
    plt.ylabel('# Images')
    plt.show()

plot_steering_histogram(augmented_steering_angles, "Augmented data", num_bins=100)
