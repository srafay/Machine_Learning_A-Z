from comet_ml import Experiment

#create an experiment with your api key
experiment = Experiment(api_key="Dz2W3DAahv0OvSAERUfhA5b7I",
                        project_name='general',
                        auto_param_logging=False)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

batch_size = 16
num_classes = 10
epochs = 5
num_nodes = 16
optimizer = 'adam'
activation = 'relu'

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#these will all get logged 
params={'batch_size':batch_size,
        'epochs':epochs,
        'layer1_type':'Dense',
        'layer1_num_nodes':num_nodes,
        'layer1_activation':activation,
        'optimizer':optimizer
}
model = Sequential()
model.add(Dense(num_nodes, activation='relu', input_shape=(784,)))
#model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#print model.summary() to preserve automatically in `Output` tab
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

#will log metrics with the prefix 'train_'
with experiment.train():
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-4,patience=3, verbose=1, mode='auto')])

#will log metrics with the prefix 'test_'
with experiment.test():
    loss, accuracy = model.evaluate(x_test, y_test)
    metrics = {
        'loss':loss,
        'accuracy':accuracy
    }
    experiment.log_multiple_metrics(metrics)

experiment.log_multiple_params(params)
experiment.log_dataset_hash(x_train) #creates and logs a hash of your data