import warnings
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.callbacks import LearningRateScheduler
    from tensorflow.keras.preprocessing.image import ImageDataGenerator


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Tut-03: Hyperparameter Search Strategy for MNIST
class ExperimentsMNIST:
    '''
    ---------------------
    experiment  [integer] min=1 max=5
    ---------------------
    Description:
        * Experiment to conduct

    ---------------------
    verbose     [integer] min=0 max=2
    ---------------------
    Description:
        * <verbose> level of fit method of the tf.keras.models
          https://www.tensorflow.org/api_docs/python/tf/keras/Model

    ---------------------
    path         [string]
    ---------------------
    Description:
        * Path to parent directory of 'models' and 'logs' where trained
          models and their logs are imported from or saved in:
          e.g. path='data'
            ** models will be saved in or imported from './data/models/'
            ** logs will be saved in './data/logs/'
            ** predictions will be saved in './data/'

    ---------------------
    import_mode [boolean]
    ---------------------
    Description:
        * If False
            1. Preprocesses the original MNIST data to generate a dataset
            2. Trains the experiment models while saving logs of training
               and validation stages for each epoch in `./<path>/logs/`
            3. Saves the trained model in `./<path>/models/`

        * If True
            ** And models for the <experiment> are found in the
               '<path>/models/' directory (else aborts):
                ** And <experiment> is less than 5:

                    return imported_models

                            imported_models:
                            ----------------
                            pandas.DataFrame(
                                index=range(number_found_models),
                                columns=[
                                    'date',  # Training date
                                    'timestamp',
                                    'name',  # Model architecture
                                    'model'  # Trained tf keras model object
                                ]
                            )

                ** And <experiment> is equal to 5:
                   evalutes the models with the default MNIST test set

                    return [tuple] (imported_models, predictions)

                            imported_models:
                            ----------------
                            pandas.DataFrame(
                                index=range(number_found_models),
                                columns=[
                                    'date',  # Training date
                                    'timestamp',
                                    'name',  # Model architecture
                                    'model'  # Trained tf keras model object
                                ]
                            )

                            predictions:
                            ------------
                            [tuple] length=number_found_models

                            predictions[i] = pandas.DataFrame(
                                index=range(number_of_test_samples),
                                columns=[
                                    '0',  # Label 0 prediction confidence
                                    ... ,
                                    '9',
                                    'pred',  # argmax of first 10 columns
                                    'label',  # y_test
                                    'image'  # x_test
                                ]
                            )
    '''

    def __init__(self, path='data'):

        # Default constants for all experiments
        self.input_shape = (28, 28, 1)
        self.activation = 'relu'  # Except output layer --> softmax
        self.optimizer = 'adam'
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.path = path
        self.path_models = f'{path}/models'
        self.path_logs = f'{path}\\logs'  # <-- Double back-slashes

        # Ensuring models path exists
        if not os.path.isdir(self.path_models):
            os.makedirs(self.path_models)

        # Experiment specific constants
        self.val_split = 0.15  # First 4 experiments validation split
        self.val_samples = 10000  # 5th experiment validation samples
        self.target_size = 200000  # 5th experiment training samples

        # Making sure experiment number is not invalid
        if experiment not in (1, 2, 3, 4, 5):
            print('ABORTED: Invalid experiment number')

        else:

            if not import_mode:

                # Do preprocessing and then train the models
                self.conduct(experiment)

            else:

                # Importing models
                imported_models = self.importModels(experiment)

                # If at least 1 model was imported
                if len(imported_models) > 0:

                    # Showing imported models
                    print('Model order by timestamp:', end='\n\n')
                    print(imported_models.iloc[:, :-1], end='\n\n')
                    self.imported_models = imported_models

                    # Evalute
                    if experiment == 5:
                        predictions = self.evaluate(imported_models)
                        self.predictions = predictions

                else:
                    msg = f'ABORTED: No E{experiment} models exist in'
                    print(f'{msg} `./{self.path_models}/`')

    def experiment(self, experiment, verbose=1, no_repeat=True):

        if experiment not in (1, 2, 3, 4, 5):
            print('ABORTED: Invalid experiment number')
            return 0

        self.verbose = verbose

        if no_repeat:
            imported_models = self.importModels(experiment)
            if len(imported_models) > 0:
                print(f'\nABORTED: E{experiment} already has trained models')
                print('\nModel order by timestamp:', end='\n\n')
                print(imported_models.iloc[:, :-1], end='\n\n')
                return imported_models

        # Initializing
        self.models = {
            'date': [],
            'timestamp': [],
            'name': [],
            'model': []
        }

        # Preprocessing
        dataset = self.preprocessing(experiment)

        # Conducting experiments
        if experiment == 1:
            self.E1(*dataset)
        elif experiment == 2:
            self.E2(*dataset)
        elif experiment == 3:
            self.E3(*dataset)
        elif experiment == 4:
            self.E4(*dataset)
        else:
            self.E5(*dataset)

        # Creating dataframe

        

    def evaluate(self, models):

        # Preprocessing
        (x_test, y_test) = self.preprocessing(0)

        # Trails
        trails = len(models)
        results = {'loss': [], 'acc': []}
        predictions = []
        models.drop(columns='date', inplace=True)
        timestamp = str(int(time.time()))

        for trail, (timestamp_old, name, model) in models.iterrows():

            # Evaluating
            self.printStart(0, trail, trails, timestamp_old, goto=False)
            (loss, acc) = model.evaluate(x_test, y_test)
            results['loss'].append(loss)
            results['acc'].append(acc)

            # Getting predictions
            pred_raw = model.predict(x_test)
            pred_argmax = np.argmax(pred_raw, axis=1)

            # Creating dataframe
            images = [image for image in x_test]
            pred_df = pd.DataFrame({'pred': pred_argmax,  # Argmax values
                                    'label': y_test,
                                    'image': images})
            pred_raw = pd.DataFrame(pred_raw, columns=range(10))
            pred_df = pd.concat([pred_raw, pred_df], axis=1)
            predictions.append(pred_df)

            # Saving predictions dataframe
            path = f'{self.path}/test-pred @{timestamp}'
            path = f'{path} {name} @{timestamp_old}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(pred_df, f)

        # Tabulating results
        models['loss'] = results['loss']
        models['acc'] = results['acc']
        models.drop(columns='model', inplace=True)
        print(models)

        print('\n---------------------')
        print('Timestamp:', timestamp)
        print('---------------------', end='\n\n')

        return predictions

    def importModels(self, experiment):

        # Initializing
        models = {
            'date': [],
            'timestamp': [],
            'name': [],
            'model': []
        }

        # Importing the models
        for _, _, files in os.walk(self.path_models):
            for file in files:
                if file.find(f'E{experiment}') > -1:

                    # Model
                    path = f'{self.path_models}/{file}'
                    model = tf.keras.models.load_model(path)
                    models['model'].append(model)

                    # Name
                    name, rest = file.split('@')
                    name = name.rstrip()
                    models['name'].append(name)

                    # Timestamp
                    timestamp = int(rest.replace('.model', ''))
                    models['timestamp'].append(timestamp)

                    # Date
                    date = datetime.datetime.fromtimestamp(timestamp)
                    date = date.strftime('%Y-%m-%d %H:%M:%S')
                    models['date'].append(date)

        # Creating the dataframe
        imported_models = pd.DataFrame(models)
        imported_models.sort_values('timestamp', inplace=True)
        imported_models.reset_index(drop=True, inplace=True)

        return imported_models

    def conduct(self, experiment):

        # Preprocessing
        dataset = self.preprocessing(experiment)

        # Initializing
        self.models = {
            'date': [],
            'timestamp': [],
            'name': [],
            'model': []
        }

        eval(f'self.E{experiment}(*dataset)')
        # https://stackoverflow.com/questions/9383740/what-does-pythons-eval-do

    def preprocessing(self, experiment):

        # Importing MNIST
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Scaling and reshaping features
        x_train = np.array(x_train).reshape((-1, 28, 28, 1)) / 255.0
        x_test = np.array(x_test).reshape((-1, 28, 28, 1)) / 255.0

        # Set 0: testing set for evalution of experiment 5 models
        if experiment == 0:

            # Class distribution: testing set
            self.printClassDistribution(y_test, 'Testing', top=True)
            return (x_test, y_test)

        # Class distribution: original training set
        self.printClassDistribution(y_train, 'Training [org]', top=True)

        # Set 1: combine training and testing sets
        if experiment < 5:
            x_train = np.concatenate((x_train, x_test), axis=0)
            y_train = np.concatenate((y_train, y_test), axis=0)

            # Class distribution: training set
            self.printClassDistribution(y_train, 'Training [comb]', end='\n\n')
            return (x_train, y_train)

        # Set 2: split and then enlarge remaining training set
        else:

            # Splitting original training set into a new smaller training set
            # and a validation set
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train,
                test_size=self.val_samples
            )

            # Class distribution: validation and training set
            self.printClassDistribution(y_val, 'Validation')
            self.printClassDistribution(y_train, 'Training [pre]')

            # Setting up the image agumentor and the agumented image generator
            agumentor = ImageDataGenerator(rotation_range=10,
                                           zoom_range=0.10,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1)
            generator = agumentor.flow(x_train, y_train,
                                       batch_size=x_train.shape[0])

            # Adding agumented images
            target_size = self.target_size
            for x_agumented_batch, y_agumented_batch in generator:  # Inf loop
                x_train = np.concatenate((x_train, x_agumented_batch))
                y_train = np.concatenate((y_train, y_agumented_batch))

                # Class distribution: training set progress
                self.printClassDistribution(y_train, 'Training [add]',
                                            end='\r')

                # Breaking the loop once target size is reached
                if y_train.shape[0] >= target_size:
                    break

            # Final agumented dataset
            self.printClassDistribution(y_train, 'Training [agu]', end='\n\n')
            return (x_train, y_train, (x_val, y_val))

    def E1(self, x, y, trails=3, epochs=15):

        # Fit params
        print('shape features ---', x.shape)
        print('shape labels -----', y.shape)
        print('epochs -----------', epochs, end='\n')

        # Trails
        for trail in range(trails):
            timestamp = str(int(time.time()))
            model = Sequential()

            # 1st conv - Input layer
            model.add(Conv2D(24, kernel_size=5, padding='same',
                             activation=self.activation,
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=2, strides=2))
            NAME = 'E1 [24C5-P2]'

            # 2nd conv
            if trail > 0:
                model.add(Conv2D(48, kernel_size=5, padding='same',
                                 activation=self.activation))
                model.add(MaxPooling2D(pool_size=2, strides=2))
                NAME = f'{NAME} - [48C5-P2]'

            # 3rd conv
            if trail > 1:
                model.add(Conv2D(64, kernel_size=5, padding='same',
                                 activation=self.activation))
                model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
                NAME = f'{NAME} - [64C5-P2]'

            # Classification layer
            model.add(Flatten())
            model.add(Dense(256, activation=self.activation))
            NAME = f'{NAME} - 256'

            # Output layer
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

            # Tensor Board
            NAME = f'{NAME} - 10 @{timestamp}'
            self.printStart(NAME[:2], trail, trails, timestamp)
            tensorboard = TensorBoard(log_dir=f'{self.path_logs}\\{NAME}',
                                      profile_batch=0)

            # Fitting
            model.fit(x, y, epochs=epochs, callbacks=[tensorboard],
                      validation_split=self.val_split,
                      verbose=self.verbose)

            # Saving
            model.save(f'{self.path_models}/{NAME}.model')

            # Appending
            date = datetime.datetime.fromtimestamp(int(timestamp))
            date = date.strftime('%Y-%m-%d %H:%M:%S')
            self.models['date'].append(date)
            self.models['timestamp'].append(int(timestamp))
            self.models['name'].append(NAME[:NAME.find(' @')])
            self.models['model'].append(model)

    def E2(self, x, y, trails=6, epochs=15):

        # Fit params
        print('shape features ---', x.shape)
        print('shape labels -----', y.shape)
        print('epochs -----------', epochs, end='\n\n')

        # Trails
        for trail in range(trails):
            timestamp = str(int(time.time()))
            model = Sequential()
            C1 = trail * 8 + 8
            C2 = trail * 16 + 16

            # 1st conv
            model.add(Conv2D(C1, kernel_size=5, activation=self.activation,
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=2, strides=2))

            # 2nd conv
            model.add(Conv2D(C2, kernel_size=5, activation=self.activation))
            model.add(MaxPooling2D(pool_size=2, strides=2))

            # Classification layer
            model.add(Flatten())
            model.add(Dense(256, activation=self.activation))

            # Output layer
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

            # Tensor Board
            NAME = f'E2 [{C1}C5-P2] - [{C2}C5-P2] - 256 - 10'
            NAME = f'{NAME}  @{timestamp}'
            self.printStart(NAME[:2], trail, trails, timestamp)
            tensorboard = TensorBoard(log_dir=f'{self.path_logs}\\{NAME}',
                                      profile_batch=0)

            timestamp = str(int(time.time()))
            self.printStart(NAME[:2], trail, trails, timestamp)

            # Fitting
            model.fit(x, y, epochs=epochs, callbacks=[tensorboard],
                      validation_split=self.val_split,
                      verbose=self.verbose)

            # Saving
            model.save(f'{self.path_models}/{NAME}.model')

            # Appending
            date = datetime.datetime.fromtimestamp(int(timestamp))
            date = date.strftime('%Y-%m-%d %H:%M:%S')
            self.models['date'].append(date)
            self.models['timestamp'].append(int(timestamp))
            self.models['name'].append(NAME[:NAME.find(' @')])
            self.models['model'].append(model)

    def E3(self, x, y, trails=6, epochs=15):

        # Fit params
        print('shape features ---', x.shape)
        print('shape labels -----', y.shape)
        print('epochs -----------', epochs, end='\n\n')

        # Trails
        for trail in range(trails):
            timestamp = str(int(time.time()))
            model = Sequential()
            D = 2 ** (trail + 4)
            if D >= 256:  # 256 already done in experiment 2
                D = 2 ** (trail + 5)

            # 1st conv
            model.add(Conv2D(32, kernel_size=5, activation=self.activation,
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=2, strides=2))

            # 2nd conv
            model.add(Conv2D(64, kernel_size=5, activation=self.activation))
            model.add(MaxPooling2D(pool_size=2, strides=2))

            # Classification layer
            model.add(Flatten())
            NAME = 'E3 [32C5-P2] - [64C5-P2]'
            if trail > 0:
                model.add(Dense(D, activation=self.activation))
                NAME = f'{NAME} - {D} - 10 @{timestamp}'
            else:
                NAME = f'{NAME} - X - 10 @{timestamp}'

            # Output layer
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

            # Tensor Board
            self.printStart(NAME[:2], trail, trails, timestamp)
            tensorboard = TensorBoard(log_dir=f'{self.path_logs}\\{NAME}',
                                      profile_batch=0)

            # Fitting
            model.fit(x, y, epochs=epochs, callbacks=[tensorboard],
                      validation_split=self.val_split,
                      verbose=self.verbose)

            # Saving
            model.save(f'{self.path_models}/{NAME}.model')

            # Appending
            date = datetime.datetime.fromtimestamp(int(timestamp))
            date = date.strftime('%Y-%m-%d %H:%M:%S')
            self.models['date'].append(date)
            self.models['timestamp'].append(int(timestamp))
            self.models['name'].append(NAME[:NAME.find(' @')])
            self.models['model'].append(model)

    def E4(self, x, y, trails=8, epochs=30):

        # Fit params
        print('shape features ---', x.shape)
        print('shape labels -----', y.shape)
        print('epochs -----------', epochs, end='\n\n')

        # Trails
        for trail in range(trails):
            timestamp = str(int(time.time()))
            model = Sequential()
            D = trail * 0.1

            NAME = f'E4 [32C5-P2-] - D{D} - [64C5-P2] - D{D}'
            NAME = f'{NAME} - 128 - D{D} - 10 @{int(time.time())}'

            # 1st conv
            model.add(Conv2D(32, kernel_size=5, activation=self.activation,
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=2, strides=2))
            model.add(Dropout(D))

            # 2nd conv
            model.add(Conv2D(64, kernel_size=5, activation=self.activation))
            model.add(MaxPooling2D(pool_size=2, strides=2))
            model.add(Dropout(D))

            # Classification layer
            model.add(Flatten())
            model.add(Dense(128, activation=self.activation))
            model.add(Dropout(D))

            # Output layer
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

            # Tensor Board
            self.printStart(NAME[:2], trail, trails, timestamp)
            tensorboard = TensorBoard(log_dir=f'{self.path_logs}\\{NAME}',
                                      profile_batch=0)

            # Fitting
            model.fit(x, y, epochs=epochs, callbacks=[tensorboard],
                      validation_split=self.val_split,
                      verbose=self.verbose)

            # Saving
            model.save(f'{self.path_models}/{NAME}.model')

            # Appending
            date = datetime.datetime.fromtimestamp(int(timestamp))
            date = date.strftime('%Y-%m-%d %H:%M:%S')
            self.models['date'].append(date)
            self.models['timestamp'].append(int(timestamp))
            self.models['name'].append(NAME[:NAME.find(' @')])
            self.models['model'].append(model)

    def E5(self, x, y, val_set, trails=2, epochs=35):

        # Shapes
        print('shape x_train ---', x.shape)
        print('shape y_train ---', y.shape)
        print('shape x_val -----', val_set[0].shape)
        print('shape y_val -----', val_set[1].shape, end='\n\n')

        # Trails
        for trail in range(trails):
            if trail == 0:
                self.modelArchitectureOld(x, y, val_set, epochs)
            elif trail == 1:
                self.modelArchitectureNew(x, y, val_set, epochs)

    def modelArchitectureOld(self, x, y, val_set, epochs):
        timestamp = str(int(time.time()))
        model = Sequential()
        D = 0.4

        # 1st conv
        model.add(Conv2D(32, kernel_size=5, activation=self.activation,
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(D))

        # 2nd conv
        model.add(Conv2D(64, kernel_size=5, activation=self.activation))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Dropout(D))

        # Classification layer
        model.add(Flatten())
        model.add(Dense(128, activation=self.activation))
        model.add(Dropout(D))

        # Output and compile
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)

        # TensorBoard
        NAME = f'E5 [32C5-P2-] - D{D} - [64C5-P2] - D{D}'
        NAME = f'{NAME} - 128 - D{D} - 10 @{timestamp}'
        self.printStart(NAME[:2], 0, 2, timestamp)
        tensorboard = TensorBoard(log_dir=f'{self.path_logs}\\{NAME}',
                                  profile_batch=0)

        # Learning rate scheduler
        annealer = LearningRateScheduler(self.exponentialDecay)

        # Fitting
        model.fit(x, y, epochs=epochs, callbacks=[tensorboard, annealer],
                  validation_data=val_set,
                  verbose=self.verbose)
        model.save(f'{self.path_models}/{NAME}.model')

        # Appending
        date = datetime.datetime.fromtimestamp(int(timestamp))
        date = date.strftime('%Y-%m-%d %H:%M:%S')
        self.models['date'].append(date)
        self.models['timestamp'].append(int(timestamp))
        self.models['name'].append(NAME[:NAME.find(' @')])
        self.models['model'].append(model)

    def modelArchitectureNew(self, x, y, val_set, epochs):
        timestamp = str(int(time.time()))
        model = Sequential()

        # 1st conv replacement
        model.add(Conv2D(32, kernel_size=3, activation=self.activation,
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation=self.activation))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=5, strides=2, padding='same',
                         activation=self.activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        # 2nd conv replaement
        model.add(Conv2D(64, kernel_size=3, activation=self.activation))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=3, activation=self.activation))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                         activation=self.activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        # Classification layer
        model.add(Flatten())
        model.add(Dense(128, activation=self.activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        # Output and compile
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)

        # TensorBoard
        NAME = 'E5 [32C3n-32C3n-32C5S2n] - D0.4 - [64C3n-64C3n-64C5S2n] - D0.4'
        NAME = f'{NAME} - 128n - D0.4 - 10 @{timestamp}'
        self.printStart(NAME[:2], 1, 2, timestamp)
        tensorboard = TensorBoard(log_dir=f'{self.path_logs}\\{NAME}',
                                  profile_batch=0)

        # Learning rate scheduler
        annealer = LearningRateScheduler(self.exponentialDecay)

        # Fitting
        model.fit(x, y, epochs=epochs, callbacks=[tensorboard, annealer],
                  validation_data=val_set,
                  verbose=self.verbose)
        model.save(f'{self.path_models}/{NAME}.model')

        # Appending
        date = datetime.datetime.fromtimestamp(int(timestamp))
        date = date.strftime('%Y-%m-%d %H:%M:%S')
        self.models['date'].append(date)
        self.models['timestamp'].append(int(timestamp))
        self.models['name'].append(NAME[:NAME.find(' @')])
        self.models['model'].append(model)

    def exponentialDecay(self, epoch):
        if epoch < 10:
            return 0.001
        else:
            # Exponential decay after 10th epoch
            return 0.001 * np.exp(0.1 * (10 - epoch))

    def printClassDistribution(self, y, title, top=False, end='\n'):
        classes, counts = np.unique(y, axis=0, return_counts=True)
        counts = counts * 100 / y.shape[0]
        for i, label in enumerate(classes):
            if i == 0:
                t = '{: 5}'.format(label)
                b = '{:5.2f}'.format(counts[i])
            else:
                t = t + '|{: 5}'.format(label)
                b = b + '|{:5.2f}'.format(counts[i])

        if top:
            print('{}| {:>7}\n{}| {: 7} {}'.format(t, 'TOTAL',
                                                  b, y.shape[0],
                                                  title), end=end)
        else:
            print('{}| {: 7} {}'.format(b, y.shape[0], title), end=end)

    def printStart(self, exp, trail, trails, timestamp, goto=True):
        print('\n----------------------------------')
        print(f'>> {exp} Model [{trail + 1}/{trails}] @{timestamp} STARTED')
        print('----------------------------------')
        if goto:
            print('>> type `tensorboard --logdir={}`'.format(
                self.path_logs.replace('\\', '/')))
            print('>> goto `http://localhost:6006/`')
            print('----------------------------------', end='\n\n')
