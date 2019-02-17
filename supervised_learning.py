from keras import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


class SupervisedLearning:

    def __init__(self):
        self.optimizer = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.train_generator = None
        self.validation_generator = None

    def import_data_from_dir(self, _train_dir: str, _valid_dir: str, _img_size: tuple):
        self.train_generator = ImageDataGenerator().flow_from_directory(
            directory=_train_dir,
            target_size=(_img_size[0], _img_size[1]),
            batch_size=30,
            class_mode='categorical')

        self.validation_generator = ImageDataGenerator().flow_from_directory(
            directory=_valid_dir,
            target_size=(_img_size[0], _img_size[1]),
            batch_size=1,
            class_mode='categorical')

    def define_optimizer(self):
        self.optimizer = Adam(lr=0.001)

    def execute_keras_model(self, _type: str,
                            _model: Sequential,
                            _optimizer=None,
                            _batch_size: int = 64,
                            _epoch: int = 5):
        """
        受け取ったモデルで学習を実行する機能
        :param _type: 'category' or 'regression'
        :param _model: Keras model.
        :param _optimizer: Keras Optimizer.
        :param _batch_size: Batch Size. Default: 64
        :param _epoch: Epoch. Default: 5
        :return: None as Error. 0 as Success.
        """
        #Train and Valid data check.
        if (self.x_train is None or self.y_train is None) or self.train_generator is None: return None
        if (self.x_valid is None or self.y_valid is None) or self.validation_generator is None: return None
        train_data_type = None
        if self.x_train is not None:
            train_data_type = 'from_list'
        elif self.train_generator is not None:
            train_data_type = 'from_dir'
        else:
            return train_data_type

        # Optimizer check.
        if _optimizer is None:
            self.define_optimizer()
        else:
            self.optimizer = _optimizer

        # Compile model.
        if _type == 'category':
            _model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if train_data_type == 'from_list':
            history = _model.fit(self.x_train, self.y_train,
                                 batch_size=_batch_size, epochs=_epoch,
                                 validation_data=(self.x_valid, self.y_valid))
        elif train_data_type == 'from_dir':
            history = _model.fit_generator(self.train_generator,
                                           batch_size=_batch_size, epochs=_epoch,
                                           validation_data=self.validation_generator,
                                           shuffle=True, verbose=1)
        else:
            print('[Warn] Invalid type.')
            return None

        # Save graph of loss.
        plt.figure(figsize=(8, 5))
        plt.ylabel('val_loss')
        plt.xlabel('epochs')
        plt.tight_layout()
        plt.plot(history.history['val_loss'], "o-", markersize=2, label='val_loss')
        plt.plot(history.history['loss'], "o-", markersize=2, label='loss')
        plt.legend()
        plt.savefig(os.path.join('..', 'loss.png'))

        # Save graph of accuracy.
        plt.figure(figsize=(8, 5))
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.tight_layout()
        plt.plot(history.history['val_acc'], "o-", markersize=2, label='val_acc')
        plt.plot(history.history['acc'], "o-", markersize=2, label='acc')
        plt.legend()
        plt.savefig(os.path.join('..', 'acc.png'))

        return 0
