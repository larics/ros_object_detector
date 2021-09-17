#!/usr/bin/env python3

import rospy
# import ros_numpy
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Bool
from ros_object_detector.msg import DetectedObjectArray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import

from sensor_msgs.msg import PointCloud2

from math import sqrt, isnan
import numpy as np
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2

from visualization_msgs.msg import MarkerArray, Marker

from PIL import ImageColor


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
        self.dist_thresh = rospy.get_param('~dist_thresh', 0.1)
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
        self.sub = rospy.Subscriber('/ros_object_detector/detected_objects', DetectedObjectArray, self.detection_callback)
        self.sub = rospy.Subscriber('/count', Bool, self.count)
        self.pub = rospy.Publisher('/pc', PointCloud2, queue_size=1)
        self.pub_peppers = rospy.Publisher('/peppers', PointStamped, queue_size=50)
        self.do_array = []
        self.do_n = 0

        self.frame_id = 0

        self.frames = 0
        self.cam_path = 0.
        self.cam_prev = None

        self.pub_markers = rospy.Publisher('/pepper_loc', MarkerArray, queue_size=1)
        rospy.spin()


    def detection_callback(self, data):
        curr_do_array = data
        curr_n = curr_do_array.n.data

        pcread = data.pc
        tform = data.camera_world_tf
        pcpub = do_transform_cloud(pcread, tform)
        self.pub.publish(pcpub)

        # local_frame = PointCloud2()
        # local_frame.points[0].x = 0.
        # local_frame.points[0].y = 0.
        # local_frame.points[0].z = 1.
        
        # global_frame = do_transform_cloud(local_frame, tform)

        cloud_points = [[0.0, 1.0, 1.0]]
        # header = std_msgs.msg.Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = 'camera_optical_frame'
        local_frame = pcl2.create_cloud_xyz32(pcread.header, cloud_points)

        global_frame = do_transform_cloud(local_frame, tform)

        gen = pcl2.read_points(global_frame, skip_nans=True, field_names=("x", "y", "z"))
        
        for p in gen:
            temp = p
        global_frame = Point()
        global_frame.x = temp[0]
        global_frame.y = temp[1]
        global_frame.z = temp[2]

        if self.cam_prev is not None:
            dist = sqrt(r_squared(self.cam_prev, global_frame))
            self.cam_path += abs(dist)
        self.cam_prev = global_frame
        self.frames += 1


        # print("\n\nframe id: ", self.frame_id)
        self.frame_id+=1

        new_p = 0
        measured_p = 0
        msg = PointStamped()
        for i in range(curr_n):
            if curr_do_array.objects[i].obj_class.data == 'pepper':
                position = curr_do_array.objects[i].position
                time_stamp = curr_do_array.header.stamp


                if not isnan(position.x):
                    msg.header.stamp = rospy.Time.now()
                    msg.point = position 
                    self.pub_peppers.publish(msg)
                    # print("in this frame: ", i)
                    
                    measured_p += 1
                    if self.do_n == 0:
                        do = DetectedObject(position, time_stamp)
                        self.do_array.append(do)
                        self.do_n += 1 
                        new_p += 1
                    else:
                        new_detection = True
                        for i in range(self.do_n):
                            if(self.do_array[i].compare(position)):
                                # print("new match")
                                self.do_array[i].add_new_detection(position, time_stamp)
                                new_detection = False
                                # break
                        if new_detection:
                            do = DetectedObject(position, time_stamp)
                            self.do_array.append(do)
                            self.do_n += 1
                            new_p += 1
                
                # print("checked all, added new: ", new_p)
                # print("out of: ", measured_p)
            # print("overall: ", self.do_n)
            # print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    def count(self,data):
        self.valid_do_array = []
        self.valid_do_n = 0
        for i in range(self.do_n):
            if (self.do_array[i].valid()):
                self.valid_do_array.append(self.do_array[i])
                self.valid_do_n += 1

        rospy.loginfo("Total number of peppers: " + str(len(self.do_array)))
        # print(self.do_array[1].position)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # cols = []
        # max_n = 0
        # max_p = self.do_array[0]
        # for pepper in self.do_array:
        #     if max_n < pepper.n_detections:
        #         max_n = pepper.n_detections
        #         max_p = pepper
        #     x,y,z = pepper.position.x, pepper.position.y, pepper.position.z
        #     d = 0.04
        #     ax.plot([x-d, x+d], [y, y], [z, z])
        #     c=plt.gca().lines[-1].get_color()
        #     cols.append(c)
        #     ax.plot([x, x], [y-d, y+d], [z, z], color=c)
        #     ax.plot([x, x], [y, y], [z-d, z+d], color=c)
        #     ax.scatter(x,y,z, color=c)

        #     for i,h in enumerate(pepper.hist):
        #         ax.scatter(h[0],h[1],h[2], marker='.', color=c, linewidths=i)
        # plt.close()

        # markerArray = MarkerArray()

        # i_pepp_m = 0

        # for i_pepp, pepper in enumerate(self.do_array):
        #     (r,g,b) = ImageColor.getcolor(cols[i_pepp], "RGB")
            
        #     r /= 255.
        #     g /= 255.
        #     b /= 255.

        #     for h in pepper.hist:
        #         marker = Marker()
        #         marker.header.frame_id = "world"
        #         marker.type = marker.SPHERE
        #         marker.action = marker.ADD
        #         marker.scale.x = 0.005
        #         marker.scale.y = 0.005
        #         marker.scale.z = 0.005
        #         marker.color.a = 1.0
        #         marker.color.r = r
        #         marker.color.g = g
        #         marker.color.b = b
        #         marker.id = i_pepp_m
        #         i_pepp_m+=1
                
        #         marker.pose.orientation.w = 1.0

        #         # print(pepper.position.x)
    
        #         marker.pose.position.x = h[0]
        #         marker.pose.position.y = h[1]
        #         marker.pose.position.z = h[2]
        #         markerArray.markers.append(marker)



        #     marker = Marker()
        #     marker.header.frame_id = "world"
        #     marker.type = marker.SPHERE
        #     marker.action = marker.ADD
        #     marker.scale.x = 0.01
        #     marker.scale.y = 0.01
        #     marker.scale.z = 0.01
        #     marker.color.a = 1.0
        #     marker.color.r = 1.0
        #     marker.color.g = 1.0
        #     marker.color.b = 1.0
        #     marker.id = i_pepp_m
        #     i_pepp_m+=1
            
        #     marker.pose.orientation.w = 1.0

        #     # print(pepper.position.x)
 
        #     marker.pose.position.x = pepper.position.x
        #     marker.pose.position.y = pepper.position.y
        #     marker.pose.position.z = pepper.position.z
        #     markerArray.markers.append(marker)

        # # print(markerArray)

        # self.pub_markers.publish(markerArray)

        print("overall path")
        print(self.cam_path)
        print("in how many frames")
        print(self.frames)
        # plt.show()


if __name__ == '__main__':
    rospy.init_node('ros_object_counter', log_level=rospy.INFO)
    try:
        counter = Counter()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')