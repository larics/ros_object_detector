#!/usr/bin/env python3

import rospy
import ros_numpy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from ros_object_detector.msg import DetectedObjectArray


class DetectedObject:
    def __init__(self, position, time_stamp):
        self.position = position
        self.n_detections = 1
        self.time_stamps = []
        self.time_stamps.append(time_stamp)
        #self.dist_thresh = rospy.get_param('~dist_thresh')
        self.dist_thresh = pow(0.02, 2)
        #self.n_thresh = rospy.get_param('~n_thresh')
        self.n_thresh = 0

    def add_new_detection(self, position, time_stamp):
        self.n_detections += 1
        self.time_stamps.append(time_stamp)
        self.position.x = (self.position.x + position.x) / 2
        self.position.y = (self.position.y + position.y) / 2
        self.position.z = (self.position.z + position.z) / 2

    def compare(self, position):
        dist = pow((self.position.x - position.x), 2) + \
               pow((self.position.y - position.y), 2) + \
               pow((self.position.z - position.z), 2)

        if dist < self.dist_thresh:
            return True
        else:
            return False

    def valid(self):
        if self.n_detections < self.n_thresh:
            return False
        else:
            return True
        # provjeravati time stamp, pa traziti da je max razlika izmedju vremena detekcije > ts_thresh?


class Counter(object):
    def __init__(self):
        self.sub = rospy.Subscriber('/ros_object_detector/detected_objects', DetectedObjectArray, self.detection_callback)
        self.sub = rospy.Subscriber('/count', Bool, self.count)
        self.do_array = []
        self.do_n = 0
        rospy.spin()

    def detection_callback(self, data):
        curr_do_array = data;
        curr_n = curr_do_array.n.data

        for i in range(curr_n):
            position = curr_do_array.objects[i].position
            time_stamp = curr_do_array.header.stamp

            if self.do_n == 0:
                do = DetectedObject(position, time_stamp)
                self.do_array.append(do)
                self.do_n += 1

            else:
                new_detection = True
                for i in range(self.do_n):
                    if(self.do_array[i].compare(position)):
                        self.do_array[i].add_new_detection(position, time_stamp)
                        new_detection = False
                        #rospy.loginfo("Detected same object multiple times")
                        break
                if new_detection:
                    do = DetectedObject(position, time_stamp)
                    self.do_array.append(do)
                    self.do_n += 1

        rospy.loginfo("Currently detecting " + str(self.do_n))

    def count(self,data):
        self.valid_do_array = []
        self.valid_do_n = 0
        for i in range(self.do_n):
            if (self.do_array[i].valid()):
                self.valid_do_array.append(self.do_array[i])
                self.valid_do_n += 1

        rospy.loginfo("Total number of objects: " + str(self.valid_do_n))


if __name__ == '__main__':
    rospy.init_node('ros_object_counter_old', log_level=rospy.INFO)
    try:
        counter = Counter()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
