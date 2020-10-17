import cv2
import scipy.io
import os
import pandas as pd
from PIL import Image
import numpy as np
import random
import argparse
from models.model import model


def get_images(batch_size, add, age, img_size=(160, 160)):
    rand = random.sample(range(0, len(add)), batch_size)
    X = []
    y = []
    for i in rand:
        image = cv2.imread(add[i])
        curr_img = cv2.resize(image, img_size, interpolation=cv2.INTER_CUBIC)
        curr_img = curr_img.astype('float64')
        """
        mean = curr_img.mean()
        stddev = curr_img.stddev()
        curr_img = (curr_img - mean)/stddev
        """
        curr_img = curr_img / 127.5
        curr_img = curr_img - 1
        curr_age = age[i]
        X.append(curr_img)
        y.append(curr_age)

    return X, y


def train(args):
    # Pre-processing
    imdbMat = scipy.io.loadmat('imdb_crop/imdb.mat')
    imdbPlace = imdbMat['imdb'][0][0]
    place = imdbPlace
    # Todo remove redundant variable
    # place = imdbMat['imdb'][0][0]
    where = 'imdb_crop'
    img_loc = []
    corr_ages = []

    for i in range(460723):
        # print(place[0][0][i])
        bYear = int(place[0][0][i] / 365)  # birth year
        # print(bYear)
        taken = place[1][0][i]  # photo taken
        # print(taken)
        path = place[2][0][i][0]
        age = taken - bYear
        img_loc.append(os.path.join(where, path))
        corr_ages.append(age)

    df = pd.DataFrame(img_loc, columns=['Image Location'])
    df['Age'] = corr_ages

    # Training
    if args.pre_trained == 'facenet':
        from models.Face_recognition import FR_model
        FR = FR_model()
        Model = model()
        Model.compile(loss='mean_absolute_error', optimizer='adam')
        # training loop
        length = len(df)
        print("length are {}".format(length))
        assert length > 0
        batch_size = args.batch_size
        n_batches = length // batch_size
        epochs = args.epochs
        iters = (int)(epochs * n_batches)
        assert iters > 0
        print("iters are {}".format(iters))
        for i in range(iters):
            X, Y = get_images(batch_size, df['Image Location'], df['Age'], (args.img_size, args.img_size))
            X = np.array(X)
            Y = np.array(Y)
            X = FR(X)

            assert X.shape == (batch_size, 128), 'expected shape {} O/p shape {}'.format((batch_size, 128), X.shape)
            history = Model.fit(X, Y, batch_size, 1, verbose=0)
            if (i + 1) % args.log_step == 0:
                print("Iters [{}/{}] Loss {} Batch size {}   ".format(i + 1, iters, history.history['loss'],
                                                                      args.batch_size))

        Model.save(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--pre_trained', type=str, default='facenet', help='pre-trained model to be usedx')
    parser.add_argument('--img_size', type=int, default=160, help='size of image to be fed to the model')
    parser.add_argument('--batch_size', type=int, default=50, help='batch s9ize to be used')
    parser.add_argument('--epochs', type=float, default=2, help='number of epochs to be used')
    parser.add_argument('--log_step', type=int, default=50, help='number of steps to be taken before logging')
    parser.add_argument('--save_path', type=str, default='Model_checkpoint',
                        help='path of dir where model is to be saved')
    args = parser.parse_args()
    train(args)
