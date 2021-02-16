#!/usr/bin/env python3

import math
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
from dodo_detector.detection import TFObjectDetector


class Detector(object):
    def __init__(self):
        self.saved_model = rospy.get_param('~saved_model', '')
        self.label_map = rospy.get_param('~label_map', '')
        self.confidence = rospy.get_param('~ssd_confidence', 0.5)
        self.image_topic = rospy.get_param('~image_topic')
        self.point_cloud_topic = rospy.get_param('~point_cloud_topic', None)
        self.camera_frame = rospy.get_param('~camera_frame')
        self.base_frame = rospy.get_param('~base_frame')

        self.tf_detector = TFObjectDetector(self.saved_model, self.label_map, self.confidence)

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('~labeled_image', Image, queue_size=100)
        self.detected_objects_pub = rospy.Publisher('~detected_objects', DetectedObjectArray, queue_size=100)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.point_cloud_sub = message_filters.Subscriber(self.point_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([self.image_sub, self.point_cloud_sub], 10)
        ts.registerCallback(self.callback)
        self.do_arrays = []

        self.rate = rospy.Rate(1000)

    def callback(self, image_message, pc_message):
        while not rospy.is_shutdown():
            try:
                transform = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, image_message.header.stamp)
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        # initialize message
        detected_object_array_msg = DetectedObjectArray()
        detected_object_array_msg.header = image_message.header
        detected_object_array_msg.image = image_message
        detected_object_array_msg.pc = pc_message
        detected_object_array_msg.camera_world_tf = transform
        self.do_arrays.append(detected_object_array_msg)

    def detect(self):
        while not rospy.is_shutdown():
            if not self.do_arrays:
                self.rate.sleep()

            else:
                for array in self.do_arrays:
                    self.do_arrays.remove(array)

                    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(array.pc)

                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(array.image, 'rgb8')
                    except CvBridgeError as e:
                        print(e)

                    marked_image, objects_pepper = self.tf_detector.from_image(cv_image)
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(marked_image, 'rgb8'))

                    # append poses of detected objects
                    objects = dict(objects_pepper.items())
                    n = 0
                    for obj_class in objects:
                        for obj_type_index, coordinates in enumerate(objects[obj_class]):
                            ymin, xmin, ymax, xmax = coordinates['box']
                            y_center = math.floor(ymax - ((ymax - ymin) / 2))
                            x_center = math.floor(xmax - ((xmax - xmin) / 2))

                            (x,y,z,_) = pc_array[y_center, x_center]

                            do_point = PointStamped()
                            do_point.point.x = x
                            do_point.point.y = y
                            do_point.point.z = z
                            do_point_world = do_transform_point(do_point,array.camera_world_tf)

                            detected_object_msg = DetectedObject()
                            detected_object_msg.position = do_point_world.point
                            detected_object_msg.ymin.data = ymin
                            detected_object_msg.ymax.data = ymax
                            detected_object_msg.xmin.data = xmin
                            detected_object_msg.xmax.data = xmax
                            detected_object_msg.obj_class.data = obj_class

                            array.objects.append(detected_object_msg)
                            n += 1

                    # publish only if an array is not empty
                    if n:
                        array.n.data = n
                        self.detected_objects_pub.publish(array)


if __name__ == '__main__':
    rospy.init_node('ros_object_detector', log_level=rospy.INFO)
    try:
        detector = Detector()
        detector.detect()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
