import numpy as np
import sys
import pprint
import time
from  TargetClassifier_lv1 import LV1_TargetClassifier
from labels import COLOR2ID
from evaluation import IMAGE_SIZE
from PIL import Image
from collections import defaultdict
from collections import namedtuple
from numba import jit

#@jit
class getSample:
    def integrateFtlb(self,features,label):
        return np.array([[i,k] for i,k in zip(features,label)])

    def grouping(self,features, target, radius):
        return np.sort(np.array(
            [[x[0][0], x[0][1], x[1]] for x in features if np.linalg.norm([x[0][0] - target[0][0], x[0][1] - target[0][1]]) <= radius]),axis=0)


if __name__ == '__main__':
    start = time.time()

    n_sample = 100

    target = LV1_TargetClassifier()
    target.load(r"D:\Data\PRMUalcon\2018\work\lv1_targets\classifier_01.png")

    features = np.array([[2 * np.random.rand() - 1, 2 * np.random.rand() - 1] for i in range(n_sample)])
    labels = target.predict(features)

    sample = getSample()
    features_list = sample.integrateFtlb(features,labels)

    for i in features_list:
        group = sample.grouping(features_list,i,0.5)


    elapsed_time = time.time() - start
    print("elapsed_time is {}".format(elapsed_time))