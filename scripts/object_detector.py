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
        self.confidence = rospy.get_param('~ssd_confidence', 0.6)
        self.image_topic = rospy.get_param('~image_topic')
        self.point_cloud_topic = rospy.get_param('~point_cloud_topic', None)
        self.camera_frame = rospy.get_param('~camera_frame')
        self.base_frame = rospy.get_param('~base_frame')
        self.use_median = rospy.get_param('~use_median', True)

        self.tf_detector = TFObjectDetector(self.saved_model, self.label_map, self.confidence)

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('~labeled_image', Image, queue_size=1)
        self.detected_objects_pub = rospy.Publisher('~detected_objects', DetectedObjectArray, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.point_cloud_sub = message_filters.Subscriber(self.point_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([self.image_sub, self.point_cloud_sub], 1)
        ts.registerCallback(self.callback)
        self.do_arrays = []

        # self.rate = rospy.Rate(10000)
        rospy.loginfo('Ready to detect!')
        rospy.spin()

    def callback(self, image_message, pc_message):
        while not rospy.is_shutdown():
            try:
                transform = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, image_message.header.stamp)
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("Error!")
                continue

        # initialize message
        detected_object_array_msg = DetectedObjectArray()
        detected_object_array_msg.header = image_message.header
        detected_object_array_msg.image = image_message
        detected_object_array_msg.pc = pc_message
        detected_object_array_msg.camera_world_tf = transform

        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_message)
        pc_array = pc_array[:,80:560]

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_message, 'rgb8')
        except CvBridgeError as e:
            rospy.loginfo('======================= error in conversion =========================')
            print(e)

        marked_image, objects_pepper = self.tf_detector.from_image(cv_image[:,80:560,:])
        rospy.loginfo('======================= detected =========================')
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(marked_image, 'rgb8'))
        rospy.loginfo('Published image!')
        rospy.loginfo('======================= after publishing image =========================')

        # append poses of detected objects
        objects = dict(objects_pepper.items())
        n = 0
        rospy.loginfo('======================= setting n to zero =========================')
        for obj_class in objects:
            for obj_type_index, coordinates in enumerate(objects[obj_class]):
                ymin, xmin, ymax, xmax = coordinates['box']
                y_center = math.floor(ymax - ((ymax - ymin) / 2))
                x_center = math.floor(xmax - ((xmax - xmin) / 2))

                not_nans = 0
                points = []
                for y_step in range(ymin, ymax):
                    for x_step in range(xmin,xmax):
                        (x,y,z,_) = pc_array[y_step, x_step]
                        if not math.isnan(x):
                            not_nans +=1
                            points.append((x,y,z))

                sorted_points = sorted(points, key=lambda x: x[2])
                index = math.floor(len(sorted_points) / 2)
                median = sorted_points[index]
                # rospy.loginfo(median)
                # total_pts = (xmax-xmin) * (ymax-ymin)
                # rospy.loginfo("Total: " + str(not_nans) + " out of " + str(total_pts) + " for class " + obj_class)


                do_point = PointStamped()

                if self.use_median:
                    (x,y,z) = median
                else:
                    (x,y,z,_) = pc_array[y_center, x_center]
                    # rospy.loginfo(pc_array[y_center, x_center])

                do_point.point.x = x
                do_point.point.y = y
                do_point.point.z = z
                do_point_world = do_transform_point(do_point,transform)
                # rospy.loginfo("Transformed")
                # rospy.loginfo(do_point_world.point.x)
                # rospy.loginfo(do_point_world.point.y)
                # rospy.loginfo(do_point_world.point.z)


                detected_object_msg = DetectedObject()
                detected_object_msg.position = do_point_world.point
                #if obj_class == "pepper":
                    # rospy.loginfo(x)
                    # rospy.loginfo(y)
                    # rospy.loginfo(z)
                    #
                    # rospy.loginfo(detected_object_msg.position.x)
                    # rospy.loginfo(detected_object_msg.position.y)
                    # rospy.loginfo(detected_object_msg.position.z)

                detected_object_msg.ymin.data = ymin
                detected_object_msg.ymax.data = ymax
                detected_object_msg.xmin.data = xmin
                detected_object_msg.xmax.data = xmax
                detected_object_msg.obj_class.data = obj_class
                detected_object_array_msg.objects.append(detected_object_msg)

                n += 1

        # publish only if an array is not empty
        if n:
            rospy.loginfo('======================= publishing array =========================')
            detected_object_array_msg.n.data = n
            self.detected_objects_pub.publish(detected_object_array_msg)



    def detect(self):
        while not rospy.is_shutdown():
            if not self.do_arrays:
                self.rate.sleep()
                #continue

            else:
                rospy.loginfo('Detecting')
                for array in self.do_arrays:
                    rospy.loginfo('======================= in detection =========================')
                    self.do_arrays.remove(array)

                    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(array.pc)

                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(array.image, 'rgb8')
                    except CvBridgeError as e:
                        rospy.loginfo('======================= error in conversion =========================')
                        print(e)

                    marked_image, objects_pepper = self.tf_detector.from_image(cv_image)
                    rospy.loginfo('======================= detected =========================')
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(marked_image, 'rgb8'))
                    rospy.loginfo('Published image!')
                    rospy.loginfo('======================= after publishing image =========================')

                    # append poses of detected objects
                    objects = dict(objects_pepper.items())
                    n = 0
                    rospy.loginfo('======================= setting n to zero =========================')
                    for obj_class in objects:
                        for obj_type_index, coordinates in enumerate(objects[obj_class]):
                            ymin, xmin, ymax, xmax = coordinates['box']
                            y_center = math.floor(ymax - ((ymax - ymin) / 2))
                            x_center = math.floor(xmax - ((xmax - xmin) / 2))

                            (x,y,z,_) = pc_array[y_center, x_center]
                            #(x,y,z) = pc_array[y_center, x_center]

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
                        rospy.loginfo('======================= publishing array =========================')
                        array.n.data = n
                        #self.detected_objects_pub.publish(array)


if __name__ == '__main__':
    rospy.init_node('ros_object_detector', log_level=rospy.INFO)
    try:
        detector = Detector()
        #detector.detect()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down')
