import os

import pandas
import pickle

import numpy as np

import cv2

import sklearn.metrics as sk

from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

class FakeModel:
    def __init__(self, bins_count, y_true, y_pred):
        self.positive_bins = np.zeros((28, bins_count), np.int32)
        self.negative_bins = np.zeros((28, bins_count), np.int32)

        self.elements = range(y_true.shape[0])

        self.y_true = y_true
        self.y_pred = y_pred

        self.bins_count = bins_count

        self.bin_size = 1.0 / bins_count
        self.bin_half_size = self.bin_size / 2.0

        self.centers = (np.array(range(bins_count)) + 0.5) / bins_count

        self.p_dist = []
        self.n_dist = []

        self.build()

    def build(self):
        for item in range(28):
            print("building class: " + str(item))

            self.build_for_class(item)
            self.build_distributions_for_class(item)

    def predict(self, input):
        result = np.zeros(input.shape)

        count = 0

        for item in input:
            for classnum in range(28):
                if item[classnum] == 1:
                    result[count, classnum] = np.random.choice(self.p_dist[classnum])
                else:
                    result[count, classnum] = np.random.choice(self.n_dist[classnum])

            count = count + 1

        return result

    def predict_with_thresholds(self, input, thresholds):
        predictions = self.predict(input)

        return (predictions[:, :] > thresholds[:]).astype(np.int32)

    def build_distributions_for_class(self, cnum):
        p_bins = self.positive_bins[cnum]
        n_bins = self.negative_bins[cnum]

        p_size = np.sum(p_bins)
        n_size = np.sum(n_bins)

        p_dist = np.zeros(p_size)
        n_dist = np.zeros(n_size)

        count = 0

        p_count = 0
        n_count = 0

        for item in self.centers:
            p_val = p_bins[count]
            n_val = n_bins[count]

            for p in range(p_val):
                p_dist[p_count] = item

                p_count = p_count + 1

            for n in range(n_val):
                n_dist[n_count] = item

                n_count = n_count + 1

            count = count + 1

        self.p_dist.append(p_dist)
        self.n_dist.append(n_dist)

    def build_for_class(self, class_num):
        p_bins = self.positive_bins[class_num]
        n_bins = self.negative_bins[class_num]

        examples = self.y_true[:, class_num]
        values = self.y_pred[:, class_num]

        count = 0

        for center in self.centers:
            for item in self.elements:
                pred = values[item]

                bins = p_bins

                if examples[item] == 0:
                    bins = n_bins

                if pred > center - self.bin_half_size and pred <= center + self.bin_half_size:
                    bins[count] = bins[count] + 1

            count = count + 1

def load_examples(fname):
    examples_data = pandas.read_csv(fname)

    result = np.zeros((len(examples_data), 28), np.int32)

    count = 0

    for item in examples_data["Target"]:
        parsed = list(map(int, item.split(" ")))

        arr = np.array(range(28), np.int32)

        example = np.isin(arr, parsed).astype(np.int32)

        result[count] = example

        count = count + 1

    return result

def load_predictions(fname):
    predicts_data = []

    with open(fname, 'rb') as f:
        predicts_data = pickle.load(f)

    return np.array(predicts_data)

def load_thresholds(fname):
    thresholds_data = []

    with open(fname, 'rb') as f:
        thresholds = pickle.load(f)

    return np.array(thresholds[0])

def draw_bin(cnum, model, isPositive, dir, name):
    root = dir + '/' + name + '/class_' + str(cnum)

    bins = []

    if isPositive:
        bins = model.positive_bins[cnum]
    else:
        bins = model.negative_bins[cnum]

    width = bins.shape[0]
    height = np.max(bins)

    img = np.zeros((height, width, 3))

    for x in range(width):
        for y in range(height):
            val = y <= bins[x]

            y1 = height - y - 1

            img[y1, x, 0] = val
            img[y1, x, 1] = val
            img[y1, x, 2] = val

    img = cv2.resize(img, (300, 300))

    os.makedirs(dir + '/' + name, exist_ok=True)

    positive = "negative"

    if isPositive:
        positive = "positive"

    print(root)

    save_img(root + '_hist_' + positive + '.png', img)

def draw_all(model, dir, name):
    #draw_bin(cnum, model, isPositive, dir, name)

    count = 0

    for item in range(model.positive_bins.shape[0]):
        draw_bin(count, model, True, dir, name)
        draw_bin(count, model, False, dir, name)

        count = count + 1

examples = load_examples("./cells/folds/holdout.csv")
predictions = load_predictions("./cells/net2/proteins.yaml.hold_out_pred.2.0")
thresholds = load_thresholds("cells/net2/proteins.yaml.tresholds.2.0")

model = FakeModel(100, examples, predictions)

print("running fake...")

fakes = model.predict_with_thresholds(examples, thresholds)

print(sk.f1_score(examples, fakes, average='macro'))

#model2 = FakeModel(100, examples, fakes)

# draw_all(model, "./cells", "hist_net_2")
#
# draw_all(model2, "./cells", "hist_net_fake_2")

#print(model.p_dist[0])

#print(model.n_dist[0])

# width = bins.shape[0]
# height = np.max(bins)
#
# img = np.zeros((height, width, 3))
#
# for x in range(width):
#     for y in range(height):
#         val = y <= bins[x]
#
#         y1 = height - y - 1
#
#         img[y1, x, 0] = val
#         img[y1, x, 1] = val
#         img[y1, x, 2] = val
#
# img = cv2.resize(img, (300, 300))
#
# save_img('./cells/hist1.png', img)
