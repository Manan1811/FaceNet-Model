import os
from PIL import Image
from models.model import model
import argparse
import numpy as np
import tensorflow as tf
import shutil


def create(args):
    if args.pre_trained == 'facenet':
        from models.Face_recognition import FR_model
        FR = FR_model()
        Model = tf.keras.models.load_model(args.save_path)

    path = args.img_dir + '/'
    names = os.listdir(path)
    Add = []
    Age = []
    for idx, i in enumerate(names, 0):
        curr_img = Image.open(path + i)
        # print(path+i)
        curr_img = curr_img.resize((args.img_size, args.img_size))
        curr_img = np.asarray(curr_img)
        curr_img = curr_img.astype('float64')
        curr_img /= 127.5
        curr_img = curr_img - 1
        X = [curr_img]
        X = np.asarray(X)
        assert X.shape == (1, args.img_size, args.img_size, 3), 'check input image shape'
        X = FR(X)
        y = Model(X)
        Add.append(i)
        Age.append(y)
        if (idx + 1) % args.log_step == 0:
            print('{} no of images predicted'.format(idx + 1))

    os.mkdir('Face-AHQ')
    # path = '/content/data/celeba_hq/train/male/'
    path = args.img_dir + '/'
    for i in range(len(Add)):
        ages = os.listdir('Face-AHQ')
        age = (int)(Age[i])
        add = path + Add[i]
        # creates folder
        if str(age) not in ages:
            os.mkdir('Face-AHQ/{}'.format(age))
        dest = 'Face-AHQ/{}/{}.png'.format(age, i)
        shutil.move(add, dest)

        if (i + 1) % args.log_step == 0:
            print('{} no of images saved'.format(i + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--pre_trained', type=str, default = 'facenet', help='pre-trained model to be used')
    parser.add_argument('--img_dir', type=str, default = 'data', help='pre-trained model to be used')
    parser.add_argument('--img_size', type=int, default = 160, help='size of image to be fed to the model')
    parser.add_argument('--log_step', type=int, default = 50, help='number of steps to be taken before logging')
    parser.add_argument('--save_path', type=str, default = 'Model_checkpoint',
                        help = 'path of dir where model is to be saved')
    args = parser.parse_args()
    create(args)