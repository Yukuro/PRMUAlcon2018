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

@jit
#CAUTION THIS IS PPRINTDEBUG METHOD
#DON'T FORGET TO ERASE
def prt(obj):
    return pprint.pprint(obj)

def getdictkey(dic,val):
    keys = [k for k,v in dic.items() if v == val]
    if keys:
        return keys[0]
    return None

def getdictkey_str(dic,val):
    keys = []
    for k,v in dic.items():
        if str(v) == str(val):
            print("K,V are {0} {1}".format(k,v))
            keys.append(k)
    #keys = [k for k,v in dic.items() if str(v) == str(val)]
    if keys:
        return keys[0]
    return None

def calccenter(group):
    sum_x = 0.0
    sum_y = 0.0
    for i in group:
        sum_x += i[0]
        sum_y += i[1]
    return np.float32(sum_x / len(group)), np.float32(sum_y / len(group))

#grouping with circle
def grouping(features,target,radius):
    return np.array([[x[0],x[1]] for x in features if np.linalg.norm([x[0]-target[0],x[1]-target[1]]) <= radius], dtype=np.float32)

'''
#grouping with box
def grouping(features,target,radius):
    return np.array([[x[0],x[1]] for x in features if x[0] >= target[0]-radius and x[0] <= target[0]+radius if x[1] >= target[1]-radius and x[1] <= target[1]+radius],dtype=np.float32)
'''

#convert INPUT data format
def convertInput_dict(group,label):
    group_dict = defaultdict(list)
    for lb,co in zip(label,group):
        group_dict[lb].append(co)

    return group_dict

def convertInput_list(group,label):
    return [[gr,lb] for gr,lb in zip(group,label)]

def centering(group,label,radius):
    #result = {}
    arearank = {}
    center = []
    label_result = []

    group_dict = convertInput_dict(group,label)
    group_list = convertInput_list(group,label)

    #search center of gravity position
    for lb in group_dict.keys():
        result = {}
        center_result = defaultdict(list)
        group_result = {}
        #print("lb are {0}".format(lb))
        for co in group_dict[lb]:
            #get centering range
            points = grouping(group_dict[lb],co,0.250)

            result[len(points)] = points

        arearank[lb] = result[max(result)]

    #calculate center of gravity and store it
    #remove arearank from group
    for lb_area in arearank.keys():
        #make center_result dictionaly.
        # this format {lb:np.array([ CENTER_X, CENTER_Y, CENTER_NUM ])}
        center_result[lb_area] = np.append(center_result[lb_area], np.array(calccenter(arearank[lb_area])))
        center_result[lb_area] = np.append(center_result[lb_area], len(arearank[lb_area]))

        for i in enumerate(arearank[lb_area]):
            group_list[0] = np.delete(group_list[0],np.where(group_list[0] == i[1])[0],0)

    for j in enumerate(group):
        label_result.append(getdictkey_str(group_dict,j[1]))

    label_result = np.array(label_result)
    
    return center_result,group,label_result

'''
def grouping(self,features):

def detectedge(self):
'''

class LV1_TargetClassifier:

    # ターゲット認識器をロード
    #   filename: ターゲット認識器を表現する画像のファイルパス
    def load(self, filename):
        self.img = Image.open(filename)

    # 入力された二次元特徴量に対し，その認識結果（クラスラベルID）を返す
    def predict_once(self, x1, x2):
        h = IMAGE_SIZE // 2
        x = max(0, min(IMAGE_SIZE - 1, np.round(h * x1 + h)))
        y = max(0, min(IMAGE_SIZE - 1, np.round(h - h * x2)))
        return COLOR2ID(self.img.getpixel((x, y)))

    # 入力された二次元特徴量の集合に対し，各々の認識結果を返す
    def predict(self, features):
        labels = np.zeros(features.shape[0])
        for i in range(0, features.shape[0]):
            labels[i] = self.predict_once(features[i][0], features[i][1])
        return np.int32(labels)

if __name__ == '__main__':
    start = time.time()
    n_samples = 100
    target = LV1_TargetClassifier()
    target.load(r"D:\Data\PRMUalcon\2018\work\lv1_targets\classifier_01.png")
    features = np.float32(np.array([[2 * np.random.rand() -1 ,2 * np.random.rand() -1] for i in range(n_samples)]))
    #pprint.pprint(len(features))
    #pprint.pprint(features)
    for i in features:
        #point = [np.float32(0.250),np.float32(0.5)]
        #get grouping range
        group = grouping(features,i,0.500)
        #pprint.pprint(len(group))
        #pprint.pprint(group)
        #time.sleep(1)

        labels = target.predict(group)
        #center_extract,group_extract = centering(group,labels,0.5)
        #pprint.pprint(labels)
        for i in range(100):
            print("="*i)
            if(i==0):
                center_extract,group_extract,label_extract = centering(group,labels,0.5)
            else:
                center_extract,group_extract,label_extract = centering(group_extract,label_extract,0.5)
            print("===CENTER_EXTRACT===")
            prt(center_extract)
            print("===GROUP_EXTRACT===")
            prt(group_extract)
            print("LEN(center_extract),LEN(group_extract) are {0} {1}".format(len(center_extract),len(group_extract)))
            print(" ")


        #pprint.pprint(center)

    elapsed_time = time.time() - start
    print("elapsed_time is {}".format(elapsed_time))
