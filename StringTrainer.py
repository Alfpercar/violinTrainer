import os
#import scipy.io
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../TFModels'))

from libs import util_matlab as umatlab
from libs import datasets, dataset_utils, utils
import tensorflow as tf
import datetime
import statistics, math

def main(winLSecs, energyBands_sr):
    resources_dir = "/Users/alfonso/code/violinDemos/ViolinTrainer/resources/Scarlett_V1/"
    models_dir = "/Users/alfonso/code/violinDemos/TFModels/models/"
    filenames = ['Silence', 'EString', 'AString', 'DString', 'GString']
    n_classes = 5
    if False:
        ds, Xs, Ys, windowSize= prepare_dataset_convnet(resources_dir, filenames, winLSecs, energyBands_sr, n_classes)
        X, Y, Y_pred = createConvNet(Xs, windowSize, n_classes)
        trainConvNet(models_dir, resources_dir, X, Y, Y_pred, ds, windowSize)
    else:
        ds, Xs, Ys, windowSize = prepare_dataset_ffw(resources_dir, filenames, winLSecs, energyBands_sr, n_classes)
        X, Y, Y_pred = create_ffw_net(Xs, windowSize, n_classes)
        train_ffwnet(models_dir, resources_dir, X, Y, Y_pred, ds, windowSize)

def prepare_dataset_ffw(resources_dir, filenames, winLSecs, energyBands_sr, n_classes):
    Xs = []
    ys = []

    for iString in range(0, len(filenames)):
        inputFile = resources_dir + filenames[iString] + '.16bit-EnergyBankFilter.txt'
        energy_bands = np.loadtxt(inputFile, skiprows=0).T
        energy_bands = (energy_bands / 120) + 1  # normalize [0-1]
        target = np.ones(energy_bands.shape[1]) * iString
        print("Preparing dataset: reading ", inputFile)

        if iString == 0:
            # We want winLSecs seconds of audio in our window
            # winLSecs = 0.05
            windowSize = int((winLSecs * energyBands_sr) // 2 * 2)
            # And we'll move our window by windowSize/2
            hopSize = windowSize // 2
            print('windowSize', windowSize)

        n_hops = (energy_bands.shape[1]) // hopSize
        n_hops = int(n_hops) - 1  # ??
        nFrames=len(target)
        for iframe in range(nFrames):
            frame = energy_bands[:, iframe]
            # frame=np.append(frame, pitch[iframe])

            Xs.append(frame)
            ys.append(int(target[iframe]))
            if iframe % 100 == 0:
                print("String:", iString, ", frame:", iframe, "/", nFrames)

    Xs = np.array(Xs)
    ys = np.array(ys)
    print(Xs.shape, ys.shape)

    #n_classes = 5  # 0--> not playing, 1,2,3,4 --> strings
    ds = datasets.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], one_hot=True, n_classes=n_classes)
    return ds, Xs, ys, windowSize

def create_ffw_net(Xs, windowSize, n_classes):
    print("Creating Network")
    tf.reset_default_graph()

    # Create the input to the network.  This is a 4-dimensional tensor!
    X = tf.placeholder(name='X', shape=(None, Xs.shape[1]), dtype=tf.float32)

    # Create the output to the network.  This is our one hot encoding of n_classes possible values
    Y = tf.placeholder(name='Y', shape=(None, n_classes), dtype=tf.float32)

    n_neurons = [100, 80, 60, 40, 20, 10]
    h1, W1 = utils.linear(x=X, n_output=n_neurons[0], name='layer1', activation=tf.nn.relu)
    h2, W2 = utils.linear(x=h1, n_output=n_neurons[1], name='layer2', activation=tf.nn.relu)
    h3, W3 = utils.linear(x=h2, n_output=n_neurons[2], name='layer3', activation=tf.nn.relu)
    h4, W4 = utils.linear(x=h3, n_output=n_neurons[3], name='layer4', activation=tf.nn.relu)
    h5, W5 = utils.linear(x=h4, n_output=n_neurons[4], name='layer5', activation=tf.nn.relu)
    h6, W6 = utils.linear(x=h5, n_output=n_neurons[5], name='layer6', activation=tf.nn.relu)
    Y_pred, W7 = utils.linear(x=h6, n_output=n_classes, name='pred', activation=tf.nn.softmax)

    return X, Y, Y_pred

def train_ffwnet(models_dir, resources_dir, X, Y, Y_pred, ds, windowSize):
    print("Training Network")
    cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred + 1e-12))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    predicted_y = tf.argmax(Y_pred, 1)
    actual_y = tf.argmax(Y, 1)
    correct_prediction = tf.equal(predicted_y, actual_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



    # Create a session and init!
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    retrain_ffwnet(X, Y, Y_pred, ds, windowSize, sess, accuracy, optimizer)


    now = datetime.datetime.now()
    save_path = saver.save(sess, resources_dir + "string_ffnet_" + now.strftime("%Y%m%d_%H%M") + ".ckpt")
    save_path = saver.save(sess, models_dir + "string_ffnet_" + now.strftime("%Y%m%d_%H%M") + ".ckpt")
    print("Model saved in file: %s" % save_path)


    return

def retrain_ffwnet(X, Y, Y_pred, ds, windowSize, sess, accuracy, optimizer):
    # Now iterate over our dataset n_epoch times
    n_epochs = 100
    batch_size = 400
    for epoch_i in range(n_epochs):
        print('Epoch: ', epoch_i)

        # Train
        this_accuracy = 0
        its = 0

        # Do our mini batches:
        for Xs_i, ys_i in ds.train.next_batch(batch_size):
            # Note here: we are running the optimizer so
            # that the network parameters train!
            this_accuracy += sess.run([accuracy, optimizer], feed_dict={
                X: Xs_i, Y: ys_i})[0]
            its += 1
            # print(this_accuracy / its)
        print('Training accuracy: ', this_accuracy / its)

        # Validation (see how the network does on unseen data).
        this_accuracy = 0
        its = 0

        # Do our mini batches:
        for Xs_i, ys_i in ds.valid.next_batch(batch_size):
            # Note here: we are NOT running the optimizer!
            # we only measure the accuracy!
            this_accuracy += sess.run(accuracy, feed_dict={
                X: Xs_i, Y: ys_i})
            its += 1
        print('Validation accuracy: ', this_accuracy / its)

# ------------- ConvNet
def prepare_dataset_convnet(resources_dir, filenames, winLSecs, energyBands_sr, n_classes):
    # ------------- prepare dataset
    Xs = []
    ys = []

    for iString in range(0, len(filenames)):
        inputFile = resources_dir + filenames[iString] + '.16bit-EnergyBankFilter.txt'
        energy_bands = np.loadtxt(inputFile, skiprows=0).T
        energy_bands= (energy_bands /120 )+1 #normalize [0-1]
        target = np.ones(energy_bands.shape[1]) * (iString)

        if iString == 0:
            # We want winLSecs seconds of audio in our window
            #winLSecs = 0.05
            windowSize = int((winLSecs * energyBands_sr) // 2 * 2)
            # And we'll move our window by windowSize/2
            hopSize = windowSize // 2
            print('windowSize', windowSize)

        n_hops = (energy_bands.shape[1]) // hopSize
        n_hops = int(n_hops) - 1        #??
        for hop_i in range(n_hops):
            # Creating our sliding window
            frames = energy_bands[:, (hop_i * hopSize):(hop_i * hopSize + windowSize)]
            avgString = round(statistics.median(target[(hop_i * hopSize):(hop_i * hopSize + windowSize)]))
            if (avgString - target[hop_i * hopSize] == 0):  #take only windows in the same string
                Xs.append(frames[..., np.newaxis])
                ys.append(int(avgString))
                #ys.append(target[(hop_i * hopSize):(hop_i * hopSize + windowSize)])

    Xs = np.array(Xs)
    ys = np.array(ys)
    print("Xs.shape:", Xs.shape, ", Xs.shape:", ys.shape)
    #ds = datasets.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], n_classes=0)
    ds = datasets.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], one_hot=True, n_classes=n_classes)

    return ds, Xs, ys, windowSize


def createConvNet(Xs, windowSize, n_classes):
    # ---------- create ConvNet
    tf.reset_default_graph()

    X = tf.placeholder(name='X', shape=(None, Xs.shape[1], Xs.shape[2], Xs.shape[3]), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_classes), dtype=tf.float32)

    # TODO:  Explore different numbers of layers, and sizes of the network
    n_filters = [20, 20, 20]

    # Now let's loop over our n_filters and create the deep convolutional neural network
    H = X
    for layer_i, n_filters_i in enumerate(n_filters):
        # Let's use the helper function to create our connection to the next layer:
        # TODO: explore changing the parameters here:
        H, W = utils.conv2d(
            H, n_filters_i, k_h=2, k_w=2, d_h=2, d_w=2,
            name=str(layer_i))

        # And use a nonlinearity
        # TODO: explore changing the activation here:
        H = tf.nn.softplus(H)

        # Just to check what's happening:
        print(H.get_shape().as_list())

    # Connect the last convolutional layer to a fully connected network
    fc, W = utils.linear(H, n_output=100, name="fcn1", activation=tf.nn.relu)
    # fc2, W = utils.linear(fc, n_output=50, name="fcn2", activation=tf.nn.relu)
    # fc3, W = utils.linear(fc, n_output=10, name="fcn3", activation=tf.nn.relu)

    # And another fully connceted network, now with just n_classes outputs, the number of outputs that our
    # one hot encoding has
    Y_pred, W = utils.linear(fc, n_output=n_classes, name="pred", activation=tf.nn.sigmoid)

    return X, Y, Y_pred



def trainConvNet(models_dir, resources_dir, X, Y, Y_pred, ds, windowSize):
    loss = tf.squared_difference(Y_pred, Y)
    cost = tf.reduce_mean(tf.reduce_sum(loss, 1))
    predicted_y = tf.argmax(Y_pred, 1)
    actual_y = tf.argmax(Y, 1)
    correct_prediction = tf.equal(predicted_y, actual_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Explore these parameters: (TODO)
    batch_size = 400

    # Create a session and init!
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    retrain_conv_net(X, Y, Y_pred, ds, windowSize, sess, batch_size, accuracy, optimizer)

    now = datetime.datetime.now()
    save_path = saver.save(sess, resources_dir + "string_convnet_" + now.strftime("%Y%m%d_%H%M") + ".ckpt")
    print("Model saved in file: %s" % save_path)
    save_path = saver.save(sess, models_dir + "string_convnet_" + now.strftime("%Y%m%d_%H%M") + ".ckpt")

    return


def retrain_conv_net(X, Y, Y_pred, ds, windowSize, sess, batch_size, accuracy, optimizer):
    # Now iterate over our dataset n_epoch times
    n_epochs = 100
    # Now iterate over our dataset n_epoch times
    for epoch_i in range(n_epochs):
        print('Epoch: ', epoch_i)

        # Train
        this_accuracy = 0
        its = 0

        # Do our mini batches:
        for Xs_i, ys_i in ds.train.next_batch(batch_size):
            # Note here: we are running the optimizer so
            # that the network parameters train!
            this_accuracy += sess.run([accuracy, optimizer], feed_dict={
                X: Xs_i, Y: ys_i})[0]
            its += 1
            # print(this_accuracy / its)
        print('Training accuracy: ', this_accuracy / its)

        # Validation (see how the network does on unseen data).
        this_accuracy = 0
        its = 0

        # Do our mini batches:
        for Xs_i, ys_i in ds.valid.next_batch(batch_size):
            # Note here: we are NOT running the optimizer!
            # we only measure the accuracy!
            this_accuracy += sess.run(accuracy, feed_dict={
                X: Xs_i, Y: ys_i})
            its += 1
        print('Validation accuracy: ', this_accuracy / its)


if __name__ == "__main__":
    winLSecs = 0.1
    energyBands_sr = 128
    main(winLSecs, energyBands_sr)