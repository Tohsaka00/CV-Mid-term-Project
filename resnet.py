import keras
import argparse
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
import random
import matplotlib.pyplot as plt

# set GPU memory
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    sess = tf.Session(config=config)


# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
                help='batch size(default: 512)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
                help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar100", metavar='STRING',
                help='dataset. (default: cifar100)')

args = parser.parse_args()

stack_n            = args.stack_n
layers             = 6 * stack_n + 2
global num_classes
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = args.batch_size
epochs             = args.epochs
iterations         = 100000 // batch_size + 1
weight_decay       = 5e-4
AUTO = tf.data.experimental.AUTOTUNE

def color_preprocessing(x_train,x_test, y_train):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    output_train = x_train
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    cutmix_train, cutmix_y = cutmix(x_train, y_train, alpha=2)
    output_train = np.vstack((cutmix_train, x_train))
    output_y = np.vstack((cutmix_y, y_train))
    return output_train, x_test, output_y

def auc(y_true, y_pred):
    auc1 = tf.metrics.auc(y_true, y_pred, curve='PR')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc1

def scheduler(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    return 0.0008


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup(data, target, alpha):
    indices = [i for i in range(data.shape[0])]
    random.shuffle(indices)
    shuffled_target = target[indices]
    output_y = []
    new_data = np.array(data)
    for i in range(len(indices)):
        lam = np.random.beta(alpha, alpha)

        new_data[i, :, :, :] = lam * data[i, :, :, :] + (1 - lam) * data[indices[i], :, :, :]
        # adjust lambda to exactly match pixel ratio

        output_y.append(lam * target[i] + (1 - lam) * shuffled_target[i])
    output_y = np.array(output_y)
    return new_data, output_y

def cutmix(data, target, alpha):
    indices = [i for i in range(data.shape[0])]
    random.shuffle(indices)
    shuffled_target = target[indices]
    output_y = []
    new_data = np.array(data)
    for i in range(len(indices)):
        lamb = np.clip(np.random.beta(alpha, alpha),0.7,0.8)
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.shape, lamb)
        new_data[i, bby1:bby2, bbx1:bbx2,:] = data[indices[i], bby1:bby2, bbx1:bbx2, :]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.shape[1] * data.shape[2] ))
        output_y.append(lam * target[i] + (1 - lam) * shuffled_target[i])
    output_y = np.array(output_y)
    return new_data, output_y


def cutout(data, target, alpha):
   indices = [i for i in range(data.shape[0])]
   random.shuffle(indices)
   shuffled_target = target[indices]
   new_data = np.array(data)
   z = np.zeros((32,32,3))
   for i in range(len(indices)):
      lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
      bbx1, bby1, bbx2, bby2 = rand_bbox(data.shape, lam)
      new_data[i, bby1:bby2, bbx1:bbx2,:] = z[bby1:bby2, bbx1:bbx2, :]
      # adjust lambda to exactly match pixel ratio
      lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.shape[-1] * data.shape[-2] ))
      targets = (target, shuffled_target, lam)
   return new_data, targets


def residual_network(img_input,classes_num=100,stack_n=5):

    def residual_block(x,o_filters,increase=False):
        stride = (1,1)
        if increase:
            stride = (2,2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x,16,False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x,64,True)
    for _ in range(1,stack_n):
        x = residual_block(x,64,False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x


if __name__ == '__main__':
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    print("========================================")
    print("MODEL: Residual Network ({:2d} layers)".format(6*stack_n+2))
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))


    print("== LOADING DATA... ==")
    # load data
    if args.dataset == "cifar100":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train, x_test, y_train = color_preprocessing(x_train, x_test, y_train)
    # show three figures after cutmix
    plt.subplot(131)
    plt.imshow(x_train[0])
    plt.subplot(132)
    plt.imshow(x_train[10])
    plt.subplot(133)
    plt.imshow(x_train[20])
    plt.show()
    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = residual_network(img_input,num_classes,stack_n)
    resnet    = Model(img_input, output)



    # print model architecture if you need.
    # print(resnet.summary())


    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    cbks = [TensorBoard(log_dir='./resnet_{:d}_{}/'.format(layers, args.dataset), histogram_freq=0),
            LearningRateScheduler(scheduler)]

    # dump checkpoint if you need.(add it to cbks)
    # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)

    datagen.fit(x_train)


    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    resnet.save('resnet_{:d}_{}.h5'.format(layers,args.dataset))