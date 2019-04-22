from scipy.misc import imread, imresize, imsave
import argparse
import os




parser = argparse.ArgumentParser(description="your script description")
parser.add_argument('--batch-size')

args = parser.parse_args()
batch_size = int(args.batch_size)

size = [224, 448, 224*4, 224*6]
dir_path = './data_source{}/'

for j in range(len(size)):
    print 'start {}'.format(size[j])
    for i in range(batch_size):
        # index = i % 481
        img1 = imread('./' + str(1) + '.jpg', mode='RGB')
        img1 = imresize(img1, (size[j], size[j]))
        if not os.path.exists(dir_path.format(size[j])):
            os.mkdir(dir_path.format(size[j]))
        imsave(dir_path.format(size[j])+'{}.jpg'.format(str(i)), img1)
    print 'end {}'.format(size[j])
