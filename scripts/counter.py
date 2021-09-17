#!/usr/bin/env python3

import rospy
# import ros_numpy
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Bool, String
from ros_object_detector.msg import DetectedObjectArray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import

from sensor_msgs.msg import PointCloud2

from math import sqrt, isnan
import numpy as np
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from visualization_msgs.msg import MarkerArray, Marker

from PIL import ImageColor

import pickle 



def r_squared(p1,p2):
    return pow((p1.x - p2.x), 2) + \
            pow((p1.y - p2.y), 2) + \
            pow((p1.z - p2.z), 2)


class DetectedObject:
    def __init__(self, position, time_stamp):
        self.position = position
        self.n_detections = 1
        self.time_stamps = []
        self.time_stamps.append(time_stamp)
        self.dist_thresh = rospy.get_param('~dist_thresh', 0.04)
        self.n_thresh = rospy.get_param('~n_thresh', 2)
        self.hist = [[position.x, position.y, position.z]]

    def add_new_detection(self, position, time_stamp):
        self.n_detections += 1
        self.time_stamps.append(time_stamp)
        # self.position.x = (self.position.x + position.x) / 2
        # self.position.y = (self.position.y + position.y) / 2
        # self.position.z = (self.position.z + position.z) / 2
        
        self.hist.append([position.x, position.y, position.z])
        x = 0
        y = 0
        z = 0
        for p in self.hist:
            x += p[0]
            y += p[1]
            z += p[2]
        self.position.x = x / self.n_detections
        self.position.y = y / self.n_detections
        self.position.z = z / self.n_detections
        

    def compare(self, position):
        dist = r_squared(self.position, position)

        if sqrt(dist) < self.dist_thresh:
            return True
        else:
            return False

    def valid(self):
        if self.n_detections < self.n_thresh:
            return False
        else:
            return True


class Counter(object):
    def __init__(self):
        sub = rospy.Subscriber('/count', Bool, self.count)
        sub = rospy.Subscriber('/plot', Bool, self.plot)
        sub = rospy.Subscriber('/save', String, self.save)
        sub = rospy.Subscriber('/peppers', PointStamped, self.pepper_cb)
        
        self.pepper_list = []

        self.do_array = []

        self.do_n = 0
        self.pub_markers = rospy.Publisher('/pepper_loc', MarkerArray, queue_size=1)
        rospy.spin()

    def pepper_cb(self, data):
        self.pepper_list.append(data)


    def plot(self,data):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cols = []
        for pepper_msg in self.pepper_list[::2]:
            pepper = pepper_msg.point
            x,y,z = pepper.x, pepper.y, pepper.z
            # x,y,z = pepper.position.x, pepper.position.y, pepper.position.z
            d = 0.04
            # ax.plot([x-d, x+d], [y, y], [z, z])
            # c=plt.gca().lines[-1].get_color()
            # cols.append(c)
            # ax.plot([x, x], [y-d, y+d], [z, z], color=c)
            # ax.plot([x, x], [y, y], [z-d, z+d], color=c)
            # ax.scatter(x,y,z, color=c)
            ax.scatter(x,y,z)

        plt.show()
        # for i,h in enumerate(pepper.hist):
        #     ax.scatter(h[0],h[1],h[2], marker='.', color=c, linewidths=i)

    def save(self,data):

        print("how many peppers: ", len(self.pepper_list))
        print("all the peppers")
        # print(self.pepper_list)

        for pl in self.pepper_list:
            print(pl)
            # print(len(pl))
            print("\n")

        if data.data=="":
            path = "peppers_loc.npz"
        else:
            path = data.data
        
        with open(path, "wb") as f:
            pickle.dump(self.pepper_list, f)


    def count(self,data):
        pass
    #             if self.do_n == 0:
    #                 do = DetectedObject(position, time_stamp)
    #                 self.do_array.append(do)
    #                 self.do_n += 1 
    #                 new_p += 1
    #             else:
    #                 new_detection = True
    #                 for i in range(self.do_n):
    #                     if(self.do_array[i].compare(position)):
    #                         print("new match")
    #                         self.do_array[i].add_new_detection(position, time_stamp)
    #                         new_detection = False
    #                         # break
    #                 if new_detection:
    #                     do = DetectedObject(position, time_stamp)
    #                     self.do_array.append(do)
    #                     self.do_n += 1
    #                     new_p += 1



if __name__ == '__main__':
    rospy.init_node('pepper_counter', log_level=rospy.INFO)
    try:
        counter = Counter()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')