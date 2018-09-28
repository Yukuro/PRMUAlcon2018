import numpy as np
import pprint
import time
from itertools import combinations
from  TargetClassifier_lv1 import LV1_TargetClassifier
from operator import itemgetter
from numba import jit

#@jit
class getSample:
    #gibe 3D LIST (NOT NDARRY)
    def unpack3Dlist(self,packed):
        unpacked = []
        for p in packed:
            for k in p:
                unpacked.append(k)
        return unpacked

    def integrateFtlb(self,features,label):
        return np.array([[i[0],i[1],k] for i,k in zip(features,label)])

    def deleteElement(self,group,element):
        for i in group:
            for k in element:
                group = np.delete(group,np.where(i[0] == k[0] and i[1] == k[1])[0],0)
        return group

    def deleteDuplicatelb(self,combination):
        for pair in combination:
            lb_ord = -1
            for coord in enumerate(pair):
                # ones counter
                if coord[0] == 0:
                    lb_ord = coord[1][2]
                    continue

                if coord[1][2] == lb_ord:
                    combination.remove(pair)
        return combination



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
            #define centering range and radius
            points = self.grouping(grouplb, coord[1][0:2], 0.5)
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

    def getCenter(self,center_coordlist,group_pointsnum,group_radius):
        result = []
        for center_group in center_coordlist:
            for center_lb in center_group:
                x_sum = 0.0
                y_sum = 0.0
                lb = 0
                for center_coord in center_lb:
                    x_sum += center_coord[0]
                    y_sum += center_coord[1]
                    lb = center_coord[2]
                x_center = x_sum / len(center_lb)
                y_center = y_sum / len(center_lb)
                radius = group_radius * (len(center_lb) / group_pointsnum)
                result.append([x_center,y_center,lb,radius])
        return result

    def getLine(self,x1,y1,x2,y2):
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a,b

    def getIntersection(self,a,b,center):
        i,k,r = center[0],center[1],center[3]
        x_p = -a*b + i + a*k + np.sqrt(-np.square(b) - 2*a*b*i - np.square(a) * np.square(i) + 2*b*k + 2*a*i*k - np.square(k) + np.square(r) + np.square(a) * np.square(r))
        x_m = -a*b + i + a*k - np.sqrt(-np.square(b) - 2*a*b*i - np.square(a) * np.square(i) + 2*b*k + 2*a*i*k - np.square(k) + np.square(r) + np.square(a) * np.square(r))
        x_list = [x_p,x_m]
        coord_list = [[x,a*x+b] for x in x_list]
        return coord_list

    #give SORTED intersection_list
    def getCircle(self,intersection_list):
        circlelist = []
        x_low, y_low = 0.0, 0.0
        for inter in enumerate(intersection_list):
            x_high,y_high = inter[1][0][0], inter[1][0][1]
            if inter[0] > 0:
                radius = np.linalg.norm([x_high - x_low, y_high - y_low]) / 2
                center_x, center_y = (x_high + x_low) / 2, (y_high + y_low) / 2
                circlelist.append([center_x,center_y,radius])
            x_low, y_low = inter[1][1][0], inter[1][1][1]
        return np.array(circlelist)

    def getEdge(self,combination,center_result):
        #get intersection when there are contact circle
        for pair in combination:
            # define all intersection
            intersection_list = []

            x1, y1 = pair[0][0], pair[0][1]
            x2, y2 = pair[1][0], pair[1][1]
            a, b = self.getLine(x1,y1,x2,y2)
            u = np.array([x2 - x1, y2 - y1])
            radius_pair = [pair[0][3],pair[1][3]]

            for p in pair:
                inter_pair = self.getIntersection(a, b, p)
                intersection_list.append(inter_pair)

            for center in center_result:
                o_x, o_y = center[0], center[1]
                radius_center = center[3]
                v = np.array([o_x - x1, o_y - y1])
                L = abs(np.cross(u, v) / np.linalg.norm(u))
                if L > 0:
                    #There are contact circle
                    if L <= radius_center:
                        inter = self.getIntersection(a,b,center)
                        intersection_list.append(inter)

            intersection_list = np.array(intersection_list)
            intersection_list.sort(axis=0)

            circle = self.getCircle(intersection_list)

        return circle

    def getRadiusall(self,edgecircle_list):
        radius = 0.0
        for circle in edgecircle_list:
            radius += circle[2]
        return radius

    def shootCircle(self,edgecircle_list,radius_all,n_edge):
        features = []
        edgecounter = 0
        for circle in edgecircle_list:
            #len_upper = int(np.trunc((circle[2] / radius_all) * n_edge))
            edgecounter += 1
            if edgecounter > n_edge:
                break

            x_lower,x_upper = circle[0] - circle[2], circle[0] + circle[2]
            y_lower,y_upper = circle[1] - circle[2], circle[1] + circle[2]
            x_features = np.random.uniform(x_lower,x_upper)
            y_features = np.random.uniform(y_lower,y_upper)

            circle_features = [[x_features,y_features]]
            features.append(circle_features)


        features = self.unpack3Dlist(features)
        #features = np.array(features)

        return features




def main():
    start = time.time()

    sample_all = 200
    sample_ratio = 0.5

    n_sample = int(sample_all * sample_ratio)
    n_edge = sample_all - n_sample
    group_radius = 0.5

    target = LV1_TargetClassifier()
    target.load(r"D:\Data\PRMUalcon\2018\work\lv1_targets\classifier_01.png")

    features = np.array([[2 * np.random.rand() - 1, 2 * np.random.rand() - 1] for i in range(n_sample)])
    labels = target.predict(features)

    sample = getSample()
    features_list = sample.integrateFtlb(features, labels)

    edgecircle_list = []

    for i in enumerate(features_list):
        print("="*60)
        print("ENTER NEXT FEATURES_LIST!!!")
        print("="*60)
        center_coordlist = []
        group = sample.grouping(features_list, i[1], group_radius)
        group_divided = sample.divideListlb((group))
        for grouplb in group_divided:
            while True:
                print(len(grouplb))
                center_coord = sample.getMaxpoint(grouplb)

                if len(center_coord) == 0:
                    break

                center_coordlist.append(center_coord)

                #center[0]: prevent out of index
                if len(center_coord) > 0:
                    grouplb = sample.deleteElement(grouplb, center_coord[0])
        print(center_coordlist)

        center_result = sample.getCenter(center_coordlist,len(group),group_radius)

        # avoid to make combinations from only one element list
        if len(center_result) > 1:
            combination = list(combinations(center_result,2))
            combination = sample.deleteDuplicatelb(combination)
            # avoid to make null list
            if len(combination) > 0:
                edge = sample.getEdge(combination,center_result)
                #unpack edge group
                for e in edge:
                    if e[0] >= -1 and e[0] <= 1 and e[1] >= -1 and e[1] <= 1:
                        edgecircle_list.append(e)
        edgecircle_list.sort(key=itemgetter(2))

        radius_all = sample.getRadiusall(edgecircle_list)

    edge_features_all = []
    while True:
        edge_features = sample.shootCircle(edgecircle_list,radius_all,n_edge)
        print(len(edge_features),n_edge)
        edge_features_all += edge_features

        n_edge = n_edge - len(edge_features)
        if (n_edge == 0):
            break
    edge_features_all = np.array(edge_features_all)

    features_all = np.vstack((features,edge_features_all))


    elapsed_time = time.time() - start
    print("elapsed_time is {}".format(elapsed_time))

if __name__ == '__main__':
    main()
