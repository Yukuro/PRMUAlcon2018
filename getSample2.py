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
        return np.array([[i[0],i[1],k] for i,k in zip(features,label)])

    def deleteElement(self,group,element):
        for i in group:
            for k in element:
                group = np.delete(group,np.where(i[0] == k[0] and i[1] == k[1])[0],0)
        return group

    def divideListlb(self,group):
        index_list = [0]
        divided_list = []
        for i in enumerate(group):
            lb_ord = group[i[0]][2]
            if(i[0]+1 <= len(group)-1):
                lb_new = group[i[0]+1][2]
            if lb_ord != lb_new:
                index_list.append(i[0]+1)
        index_list.append(len(group))
        if(index_list[0] == 0 and index_list[1] == 0):
            index_list.pop(0)

        for k in enumerate(index_list):
            if(k[0]+1 < len(index_list)):
                divided_list.append(group[index_list[k[0]]:index_list[k[0]+1]])

        return divided_list

    def grouping(self,features, target, radius):
        return np.sort(np.array(
            [[x[0], x[1], x[2]] for x in features if np.linalg.norm([x[0] - target[0], x[1] - target[1]]) <= radius]),axis=0)

    def getMaxpoint(self,grouplb):
        pointsrank = []
        findedpoints = []
        #prevent maxpoint before assginment
        maxpoint_counter = 0
        for coord in enumerate(grouplb):
            lb = coord[1][2]
            points = self.grouping(grouplb, coord[1][0:2], 0.25)
            #ones counter
            if coord[0] == 0:
                len_tmp = -1

            #ones counter avoided
            if len_tmp <= len(points):
                maxpoint = points
                maxpoint_counter = len(points)
                len_tmp = len(points)

        if maxpoint_counter >= 3:
            pointsrank.append(maxpoint)

        return np.array(pointsrank)


def main():
    start = time.time()

    n_sample = 100

    target = LV1_TargetClassifier()
    target.load(r"D:\Data\PRMUalcon\2018\work\lv1_targets\classifier_01.png")

    features = np.array(
        [[2 * np.random.rand() - 1, 2 * np.random.rand() - 1] for i in range(n_sample)])
    labels = target.predict(features)

    sample = getSample()
    features_list = sample.integrateFtlb(features, labels)

    for i in enumerate(features_list):
        print(i[0])
        print("="*60)
        print("ENTER NEXT FEATURES_LIST!!!")
        print("="*60)
        center_coordlist = []
        group = sample.grouping(features_list, i[1], 0.5)
        group_divided = sample.divideListlb((group))
        for grouplb in group_divided:
            while True:
                print(len(grouplb))
                center = sample.getMaxpoint(grouplb)
                center_coordlist.append(center)

                if len(center) == 0:
                    break

                #center[0]: prevent out of index
                if len(center) > 0:
                    grouplb = sample.deleteElement(grouplb, center[0])
        print(center_coordlist)


    elapsed_time = time.time() - start
    print("elapsed_time is {}".format(elapsed_time))

if __name__ == '__main__':
    main()