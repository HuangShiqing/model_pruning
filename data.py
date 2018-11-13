import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  # dealing with plots
import tensorlayer as tl

from varible import *


def resize_img(img):
    """
        保持长宽比缩放图像到416*416大小，空余部分填128

        Parameters
        ----------
        img : np.array  [h,w,3]

        Returns
        -------
        im_sized : np.array  [416,416,3]

        Examples
        --------
    """
    img_w = img.shape[1]
    img_h = img.shape[0]

    ratio = img_w / img_h
    net_w, net_h = 224, 224
    if ratio < 1:
        new_h = int(net_h)
        new_w = int(net_h * ratio)
    else:
        new_w = int(net_w)
        new_h = int(net_w / ratio)
    im_sized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    dx = net_w - new_w
    dy = net_h - new_h

    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (int(dx / 2), 0), (0, 0)), mode='constant', constant_values=128)
        im_sized = np.pad(im_sized, ((0, 0), (0, dx - int(dx / 2)), (0, 0)), mode='constant', constant_values=128)
    else:
        im_sized = im_sized[:, -dx:, :]
    if dy > 0:
        im_sized = np.pad(im_sized, ((int(dy / 2), 0), (0, 0), (0, 0)), mode='constant', constant_values=128)
        im_sized = np.pad(im_sized, ((0, dy - int(dy / 2)), (0, 0), (0, 0)), mode='constant', constant_values=128)
    else:
        im_sized = im_sized[-dy:, :, :]
    return im_sized


def random_flip(image, flip):
    if flip == 1:
        return cv2.flip(image, 1)
    return image


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    def _rand_scale(scale):
        scale = np.random.uniform(1, scale)
        return scale if (np.random.randint(2) == 0) else 1. / scale

    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)
    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp
    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180
    # avoid overflow when astype('uint8')
    image[...] = np.clip(image[...], 0, 255)
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def read_data(data_path, valid_proportion, test_proportion, pos_path="1/", neg_path="0/"):
    """
    Args:
        data_path: list,数据所在文件夹，最后有一杠
        valid_proportion: float，验证集所占百分比，小数，如0.1
        test_proportion: float，测试集所占百分比
        pos_path: str,正样本所在文件夹
        neg_path: str,负样本所在文件夹
    Returns:
        x_train: list，dtype=str，图片路径
        y_train: list, dtype=int
        x_valid: list，dtype=str，图片路径
        y_valid: list, dtype=int
        x_test: list，dtype=str，图片路径
        y_test: list, dtype=int
    """

    pos_image_path = []
    pos_labels = []

    neg_image_path = []
    neg_labels = []

    ful_image_path = []
    ful_labels = []

    np.random.seed(0)

    pos_path = data_path + pos_path
    for img in os.listdir(pos_path):
        label = 1

        path = os.path.join(pos_path, img)
        pos_image_path.append(path)
        pos_labels.append(label)

    neg_path = data_path + neg_path
    for img in os.listdir(neg_path):
        label = 0

        path = os.path.join(neg_path, img)
        neg_image_path.append(path)
        neg_labels.append(label)

    ful_image_path = pos_image_path + neg_image_path
    ful_labels = pos_labels + neg_labels

    temp = np.array([ful_image_path, ful_labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    ful_image_path = list(temp[:, 0])
    ful_labels = list(temp[:, 1])
    ful_labels = [int(i) for i in ful_labels]

    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    from sklearn.model_selection import train_test_split
    if not valid_proportion == 0:
        x_train, x_valid, y_train, y_valid = train_test_split(ful_image_path, ful_labels,
                                                              test_size=(valid_proportion + test_proportion),
                                                              stratify=ful_labels, random_state=1)
        if not test_proportion == 0:
            x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=test_proportion / (
                    valid_proportion + test_proportion), stratify=y_valid, random_state=1)
    else:
        x_train = ful_image_path
        y_train = ful_labels

    print("train_num: %d ,pos_num: %d , neg_num: %d" % (
        len(y_train), y_train.count(1), len(y_train) - y_train.count(1)))
    print("valid_num: %d ,pos_num: %d , neg_num: %d" % (
        len(y_valid), y_valid.count(1), len(y_valid) - y_valid.count(1)))
    print("test_num : %d ,pos_num: %d , neg_num: %d" % (
        len(y_test), y_test.count(1), len(y_test) - y_test.count(1)))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_data(img_abs_path, y):
    image = cv2.imread(img_abs_path)
    image = image[:, :, ::-1]  # RGB image
    image = resize_img(image)
    image = random_distort_image(image, hue=10)

    flip = np.random.randint(2)
    image = random_flip(image, flip)

    image = tl.prepro.shift(image, is_random=True)

    return image, y


def data_generator(x_train, y_train, is_show=False):
    batch_size = Gb_batch_size
    n = len(y_train)
    i = 0
    count = 0
    while count < (n / batch_size):
        x_datas = []
        y_datas = []
        while len(y_datas) < batch_size:
            # for t in range(batch_size):
            i %= n
            x_data, y_data = get_data(x_train[i], y_train[i])
            i += 1
            if is_show == True:
                print(y_data)
                plt.cla()
                plt.imshow(x_data)
                plt.show()

            x_datas.append(x_data)
            y_datas.append(y_data)

        x_datas = np.array(x_datas)
        x_datas = x_datas / 255.
        # boxes_labeled = np.array(boxes_labeled)
        yield x_datas, y_datas
        count += 1


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test = read_data('/home/hsq/DeepLearning/data/dogVscat/train', 0.3, 0,
                                                                   pos_path="/dog/", neg_path="/cat/")
    a = data_generator(x_train, y_train, is_show=True)
    for x in a:
        print('ok')
    exit()
