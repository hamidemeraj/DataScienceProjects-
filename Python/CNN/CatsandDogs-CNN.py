# CNN with Cats and Dogs 

# Builidng CNN 
# Import Libraries 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten 
from keras.layers import Dense

# Initializing the CNN 
classifier  = Sequential()

# Step1: Convolution 
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation='relu'))

# Step2: Maxpooling
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Add Second Convolutional Layer (It is optional)
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPool2D(pool_size = (2, 2)))

# Step3: Flattening
classifier.add(Flatten())

# Step4: FullyConnection
classifier.add(Dense(128, activation = 'relu' ))
classifier.add(Dense(1, activation = 'sigmoid' ))

# Compiling the CNN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Fitting the CNN to the images 
# Image Augmentation is necessary because of avoiding fro overfitting
# Preprocessing the Training set
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('C:/Users/snapp/Data-Projects/Datasets/cats&dogs/training_set/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('C:/Users/snapp/Data-Projects/Datasets/cats&dogs/test_set/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                    sample_per_epoch = 8000,
                    nb_epoch = 25, 
                    validation_data = test_set, 
                    nb_val_samples = 2000
                    )
