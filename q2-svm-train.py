from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import joblib
import os
import glob
from skimage.feature import hog

feat_path = 'data/train_dataset/cup/features'
feat_pos_path = os.path.join(feat_path, 'pos')
feat_neg_path = os.path.join(feat_path, 'neg')

train_feat_pos_lists = glob.glob(os.path.join(feat_pos_path, '*.feat'))
train_feat_neg_lists = glob.glob(os.path.join(feat_neg_path, '*.feat'))

X = []
y = []
# 加载正例样本
for feat_pos in train_feat_pos_lists:
    feat_pos_data = joblib.load(feat_pos)
    X.append(feat_pos_data)
    y.append(1)
#     print('feat_pos_data.shape:', feat_pos_data.shape)
# 加载负例样本
for feat_neg in train_feat_neg_lists:
    feat_neg_data = joblib.load(feat_neg)
    X.append(feat_neg_data)
    y.append(0)
#     print('feat_neg_data.shape:', feat_neg_data.shape)

clf = LinearSVC()

clf.fit(X, y)

model_path = 'data/models'
joblib.dump(clf, os.path.join(model_path, 'svm_cup.model'))