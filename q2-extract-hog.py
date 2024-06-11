import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import joblib
import os
import glob
from skimage.feature import hog
import cv2

train_dataset_pos_path = os.path.expanduser('data/train_dataset/pos')
train_dataset_neg_path = os.path.expanduser('data/train_dataset/neg')

feat_path = 'data/train_dataset/cup/features'
feat_pos_path = os.path.join(feat_path, 'pos')
feat_neg_path = os.path.join(feat_path, 'neg')

train_dataset_pos_lists = glob.glob(os.path.join(train_dataset_pos_path, '*'))
train_dataset_neg_lists = glob.glob(os.path.join(train_dataset_neg_path, '*'))

# pos hog特征存储
for pos_path in train_dataset_pos_lists:
    pos_im = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
    pos_hog = hog(pos_im)
    feat_pos_name = os.path.splitext(os.path.basename(pos_path))[0] + '.feat'
    joblib.dump(pos_hog, os.path.join(feat_pos_path, feat_pos_name))

# neg hog特征存储
for neg_path in train_dataset_neg_lists:
    neg_im = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
    neg_hog = hog(neg_im)
    feat_neg_name = os.path.splitext(os.path.basename(neg_path))[0] + '.feat'
    joblib.dump(neg_hog, os.path.join(feat_neg_path, feat_neg_name))
