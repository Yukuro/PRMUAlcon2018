# coding: UTF-8

import sys
import numpy as np
import time
import imp
from PIL import Image
from sklearn import neighbors
from labels import COLOR2ID
from evaluation import IMAGE_SIZE
from evaluation import LV1_Evaluator
from getSample import getSample
from itertools import combinations
from operator import itemgetter

# ターゲット認識器を表現するクラス
# ターゲット認識器は2次元パターン（512x512の画像）で与えられるものとする
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

# クローン認識器を表現するクラス
# このサンプルコードでは単純な 1-nearest neighbor 認識器とする（sklearnを使用）
# 下記と同型の fit メソッドと predict メソッドが必要
class LV1_UserDefinedClassifier:

    def __init__(self):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=1)

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        self.clf.fit(features, labels)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        labels = self.clf.predict(features)
        return np.int32(labels)

# ターゲット認識器に入力する二次元特徴量をサンプリングする関数
#   n_samples: サンプリングする特徴量の数
def LV1_user_function_sampling(sample_all):
    start = time.time()

    sample_ratio = 0.7

    n_sample = int(sample_all * sample_ratio)
    n_edge = sample_all - n_sample
    group_radius = 0.7

    target = LV1_TargetClassifier()
    target.load(sys.argv[1])

    features = np.array(
        [[2 * np.random.rand() - 1, 2 * np.random.rand() - 1] for i in range(n_sample)])
    labels = target.predict(features)

    sample = getSample()
    features_list = sample.integrateFtlb(features, labels)

    edgecircle_list = []

    for i in enumerate(features_list):
        center_coordlist = []
        group = sample.grouping(features_list, i[1], group_radius)
        group_divided = sample.divideListlb((group))
        for grouplb in group_divided:
            while True:
                center_coord = sample.getMaxpoint(grouplb)
                if len(center_coord) == 0:
                    break

                center_coordlist.append(center_coord)



                # center[0]: prevent out of index
                if len(center_coord) > 0:
                    grouplb = sample.deleteElement(grouplb, center_coord[0])

        center_result = sample.getCenter(center_coordlist, len(group), group_radius)


        # avoid to make combinations from only one element list
        if len(center_result) > 1:
            combination = list(combinations(center_result, 2))
            combination = sample.deleteDuplicatelb(combination)
            # avoid to make null list
            if len(combination) > 0:

                edge = sample.getEdge(combination, center_result)
                # unpack edge group
                for e in edge:
                    if e[0] >= -1 and e[0] <= 1 and e[1] >= -1 and e[1] <= 1:
                        edgecircle_list.append(e)
        edgecircle_list.sort(key=itemgetter(2))

        radius_all = sample.getRadiusall(edgecircle_list)

    edge_features_all = []
    while True:
        edge_features = sample.shootCircle(edgecircle_list, radius_all, n_edge)
        edge_features_all += edge_features

        n_edge = n_edge - len(edge_features)
        if (n_edge == 0):
            break
    edge_features_all = np.array(edge_features_all)

    features_all = np.vstack((features, edge_features_all))

    elapsed_time = time.time() - start
    print("elapsed_time is {}".format(elapsed_time))

    return features_all


# クローン処理の実行
# 第一引数でターゲット認識器を表す画像ファイルのパスを，
# 第二引数でクローン認識器の可視化結果を保存する画像ファイルのパスを，
# それぞれ指定するものとする
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("usage: clone.py /target/classifier/image/path /output/image/path")
        exit(0)

    # ターゲット認識器を用意
    target = LV1_TargetClassifier()
    target.load(sys.argv[1]) # 第一引数で指定された画像をターゲット認識器としてロード
    print("\nA target recognizer was loaded from {0} .".format(sys.argv[1]))

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    n = 300
    features = LV1_user_function_sampling(sample_all=n)
    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々に対応するクラスラベルIDを取得
    labels = target.predict(features)
    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV1_UserDefinedClassifier()
    model.fit(features, labels)
    print("\nA clone recognizer was trained.")

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV1_Evaluator()
    evaluator.visualize(model, sys.argv[2])
    print("\nThe clone recognizer was visualized and saved to {0} .".format(sys.argv[2]))
    print("\naccuracy: {0}".format(evaluator.calc_accuracy(target, model)))
