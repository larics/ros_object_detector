#!/usr/bin/env python

import math
import tf
import tf2_ros
import rospy
import ros_numpy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, PoseArray, PointStamped
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
        self.camera_frame = rospy.get_param('~camera_frame', 'pepper')
        self.base_frame = rospy.get_param('~base_frame', 'pepper')

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('~labeled_image', Image, queue_size=10)
        self.pose_pub = rospy.Publisher('~detected_objects', PoseArray, queue_size=10)
        self.detector = SingleShotDetector(self.frozen_graph, self.label_map, confidence=self.confidence)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def run(self):
        while not rospy.is_shutdown():
            # read rgb image and point cloud
            image_message = rospy.wait_for_message(self.image_topic, Image)
            pc_message = rospy.wait_for_message(self.point_cloud_topic, PointCloud2)

            while not rospy.is_shutdown():
                try:
                    transform = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rospy.Time())
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue

            pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_message)

            # convert from imgmsg to cv2, detect objects and publish labeled image
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_message, 'rgb8')
                marked_image, objects = self.detector.from_image(cv_image)
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(marked_image, 'rgb8'))
            except CvBridgeError as e:
                print(e)

            # initialize message
            detected_object_array = PoseArray()
            detected_object_array.header = image_message.header

            # append poses of detected objects of type 'goal class'
            if self.goal_class in objects:
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

                        detected_object = Pose()
                        detected_object.position = do_point_world.point
                        detected_object_array.poses.append(detected_object)

                self.pose_pub.publish(detected_object_array)

            #rospy.sleep(2)

if __name__ == '__main__':
    rospy.init_node('ros_object_detector', log_level=rospy.INFO)
    try:
        detector = Detector()
        detector.run()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
