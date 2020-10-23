#!/usr/bin/env python

import math
import tf
import tf2_ros
import rospy
import ros_numpy
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from ros_object_detector.msg import DetectedObject, DetectedObjectArray
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from dodo_detector.detection import SingleShotDetector

class Detector(object):
    def __init__(self):
        self.frozen_graph = rospy.get_param('~inference_graph', '')
        self.label_map = rospy.get_param('~label_map', '')
        self.confidence = rospy.get_param('~ssd_confidence', 0.5)
        self.image_topic = rospy.get_param('~image_topic')
        self.point_cloud_topic = rospy.get_param('~point_cloud_topic', None)
        self.goal_class = rospy.get_param('~goal_class', 'pepper')
        self.camera_frame = rospy.get_param('~camera_frame')
        self.base_frame = rospy.get_param('~base_frame')

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('~labeled_image', Image, queue_size=10)
        self.detected_objects_pub = rospy.Publisher('~detected_objects', DetectedObjectArray, queue_size=10)
        self.detector = SingleShotDetector(self.frozen_graph, self.label_map, confidence=self.confidence)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.point_cloud_sub = message_filters.Subscriber(self.point_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([self.image_sub, self.point_cloud_sub], 10)
        ts.registerCallback(self.callback)
        rospy.spin()


    def callback(self, image_message, pc_message):
        time_begin1 = rospy.Time.now()
        while not rospy.is_shutdown():
            try:
                transform = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rospy.Time())
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_message)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_message, 'rgb8')
            marked_image, objects = self.detector.from_image(cv_image)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(marked_image, 'rgb8'))
        except CvBridgeError as e:
            print(e)

        # initialize message
        detected_object_array_msg = DetectedObjectArray()
        detected_object_array_msg.header = image_message.header

        # append poses of detected objects of type 'goal class'
        if self.goal_class in objects:
            n = 0
            for obj_type_index, coordinates in enumerate(objects[self.goal_class]):
                ymin, xmin, ymax, xmax = coordinates['box']
                y_center = ymax - ((ymax - ymin) / 2)
                x_center = xmax - ((xmax - xmin) / 2)

                (x,y,z,_) = pc_array[y_center, x_center]
                if not math.isnan(x):
                    do_point = PointStamped()
                    do_point.header = image_message.header
                    do_point.point.x = x
                    do_point.point.y = y
                    do_point.point.z = z
                    do_point_world = do_transform_point(do_point,transform)

                    detected_object_msg = DetectedObject()
                    detected_object_msg.position = do_point_world.point
                    detected_object_msg.ymin.data = ymin
                    detected_object_msg.ymax.data = ymax
                    detected_object_msg.xmin.data = xmin
                    detected_object_msg.xmax.data = xmax
                    detected_object_array_msg.objects.append(detected_object_msg)
                    n += 1

            # publish only if an array is not empty
            if n:
                detected_object_array_msg.n.data = n
                self.detected_objects_pub.publish(detected_object_array_msg)

        #time_end = rospy.Time.now()
        #duration = time_end - time_begin1
        #rospy.loginfo("It took me " + str(duration.to_sec()) + " secs, so my frequency is " + str(1/duration.to_sec()) + " Hz.")

if __name__ == '__main__':
    rospy.init_node('ros_object_detector', log_level=rospy.INFO)
    try:
        detector = Detector()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
