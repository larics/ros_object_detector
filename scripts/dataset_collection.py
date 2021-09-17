#!/usr/bin/env python3

import math
import tf2_ros
import rospy
import ros_numpy
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped, TransformStamped
from ros_object_detector.msg import DetectedObject, DetectedObjectArray
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


import tf

class Collector(object):
    def __init__(self):
        self.image_topic = rospy.get_param('~image_topic')
        self.point_cloud_topic = rospy.get_param('~point_cloud_topic', None)
        self.camera_frame = rospy.get_param('~camera_frame')
        self.base_frame = rospy.get_param('~base_frame')
        self.use_median = rospy.get_param('~use_median', True)

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.save_flag = False

        # self.detected_objects_pub = rospy.Publisher('~detected_objects', DetectedObjectArray, queue_size=1)
        self.image_pub = rospy.Publisher('~image_save', Image, queue_size=1)
        self.pc_pub = rospy.Publisher('~pc_save', PointCloud2, queue_size=1)
        self.tf_pub = rospy.Publisher('~tf_save', TransformStamped, queue_size=1)

        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.point_cloud_sub = message_filters.Subscriber(self.point_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([self.image_sub, self.point_cloud_sub], 1)
        ts.registerCallback(self.callback)

        self.save_service = rospy.Subscriber('/ready', Bool, self.flag_callback, queue_size=1)        

        rospy.wait_for_service('position_reached')
        self.pos_reached_srv = rospy.ServiceProxy('position_reached', SetBool)


        # self.rate = rospy.Rate(10000)
        rospy.loginfo('Ready to collect data!')
        rospy.spin()


    def callback(self, image_message, pc_message):


        if self.save_flag:
    
            rospy.loginfo("Sync image pc!")
    
            while not rospy.is_shutdown():
                try:
                    transform = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, image_message.header.stamp)
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.loginfo("Error in transform lookup!")
                    rospy.loginfo(e)
                    return

            # detected_object_array_msg = DetectedObjectArray()
            # detected_object_array_msg.header = image_message.header
            # detected_object_array_msg.image = image_message
            # detected_object_array_msg.pc = pc_message
            # detected_object_array_msg.camera_world_tf = transform

            # pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_message)
            # pc_array = pc_array[:,80:560]

            # try:
            #     cv_image = self.bridge.imgmsg_to_cv2(image_message, 'rgb8')
            # except CvBridgeError as e:
            #     rospy.loginfo('======================= error in conversion =========================')
            #     print(e)

            # self.detected_objects_pub.publish(detected_object_array_msg)

            # tf_msg = TransformStamped()
            # tf_msg.header.stamp=rospy.Time.now()
            # tf_msg.child_frame_id = self.camera_frame
            # tf_msg.transform = transform

            from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
            pc_trans = do_transform_cloud(pc_message, transform)

            self.image_pub.publish(image_message)
            self.pc_pub.publish(pc_message)
            # self.pc_pub.publish(pc_trans)
            self.tf_pub.publish(transform)

            self.pos_reached_srv(True)

            rospy.loginfo("PUBLISHED!")

            self.save_flag = False
        else:
            return


    def flag_callback(self, msg):
        print(">>>>>>>> received record flag <<<<<<<")
        self.save_flag  = True


if __name__ == '__main__':
    rospy.init_node('ros_data_collector', log_level=rospy.INFO)
    try:
        collector = Collector()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
