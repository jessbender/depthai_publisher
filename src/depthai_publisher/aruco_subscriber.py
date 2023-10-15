#!/usr/bin/env python3

import cv2
import tf2_ros
import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import Bool, Time
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PoseStamped

import numpy as np


class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/depthai_node/image/compressed'

    def __init__(self):
        #TODO: aruco target number 
        self.param_aruco_target = 49
        self.time_finished_processing = rospy.Time(0)
        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        self.land_pub = rospy.Publisher('landing_site', Bool, queue_size=2)
        self.landing = False

        self.aruco_pose_pub = rospy.Publisher('/depthai_node/detection/aruco_pose', PoseStamped, queue_size=10)
        self.pub_found_aruco = rospy.Publisher('/uavasr/aruco_found', Time, queue_size=1)

        self.sub_info = rospy.Subscriber("/depthai_node/camera/camera_info", CameraInfo, self.callback_info)
        # Set additional camera parameters
        self.got_camera_info = False
        self.camera_matrix = None
        self.dist_coeffs = None

        # TODO: change back for flight
        # self.sub_uav_pose = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.callback_uav_pose)
        self.sub_uav_pose = rospy.Subscriber('/uavasr/pose', PoseStamped, self.callback_uav_pose)

        self.br = CvBridge()

        # init UAV pose
        self.uav_pose = []
        self.x_p = "-1"
        self.y_p = "-1"
        self.target_name = ""

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)
        
        self.tfbr = tf2_ros.TransformBroadcaster()

    # Collect in the camera characteristics
    def callback_info(self, msg_in):
        self.dist_coeffs = np.array([[msg_in.D[0], msg_in.D[1], msg_in.D[2], msg_in.D[3], msg_in.D[4]]], dtype="double")

        self.camera_matrix = np.array([
                    (msg_in.P[0], msg_in.P[1], msg_in.P[2]),
                    (msg_in.P[4], msg_in.P[5], msg_in.P[6]),
                    (msg_in.P[8], msg_in.P[9], msg_in.P[10])],
                    dtype="double")

        if not self.got_camera_info:
            rospy.loginfo("Got camera info")
            self.got_camera_info = True

    def img_callback(self, msg_in):
        if msg_in.header.stamp > self.time_finished_processing:
            if self.got_camera_info:
                try:
                    frame = self.br.compressed_imgmsg_to_cv2(msg_in)
                except CvBridgeError as e:
                    rospy.logerr(e)

                if self.landing == False:
                    aruco = self.find_aruco(frame, msg_in)
                    self.publish_to_ros(aruco)

            # cv2.imshow('aruco', aruco)
            # cv2.waitKey(1)
            self.time_finished_processing = rospy.Time(0)

    def callback_uav_pose(self, msg_in):
        self.current_location = msg_in.pose.position
        self.uav_pose = [self.current_location.x, self.current_location.y, self.current_location.z, 0.0]
        self.x_p = self.uav_pose[0]
        self.y_p = self.uav_pose[1]
          

    def find_aruco(self, frame, msg_in):
        if self.got_camera_info and self.landing is False:
            (corners, ids, _) = cv2.aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.aruco_params)

            if len(corners) > 0:
                ids = ids.flatten()

                for (marker_corner, marker_ID) in zip(corners, ids):
                    if marker_ID == self.param_aruco_target:
                        aruco_corners = corners
                        corners = marker_corner.reshape((4, 2))
                        (top_left, top_right, bottom_right, bottom_left) = corners

                        top_right = (int(top_right[0]), int(top_right[1]))
                        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                        top_left = (int(top_left[0]), int(top_left[1]))

                        cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                        cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                        cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                        cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                        cv2.putText(frame, str(
                            marker_ID), (top_left[0], top_right[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Estimate the pose of the ArUco marker (not for angle only for coordinates)
                        # TODO ArUCO 0.2
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, 0.2, self.camera_matrix, self.dist_coeffs)
                        
                        if tvec is not None:
                            self.landing = True
                            rospy.loginfo("UAV, at x: {}, y: {}, z: {}".format(np.round(self.current_location.x, 2), np.round(self.current_location.y, 2), np.round(self.current_location.z, 2)))                            
                            rospy.loginfo('Landing Site: x: {}, y: {}'.format(np.round(tvec[0,0,0], 2), np.round(tvec[0,0,1], 2)))

                            msg_out = TransformStamped()
                            msg_out.header = msg_in.header
                            msg_out.child_frame_id = "aruco"
                            msg_out.transform.translation.x = -tvec[0,0,1]
                            msg_out.transform.translation.y = tvec[0,0,0]
                            msg_out.transform.translation.z = tvec[0,0,2]
                            
                            msg_out.transform.rotation.w = 1.0	
                            msg_out.transform.rotation.x = 0.0
                            msg_out.transform.rotation.y = 0.0
                            msg_out.transform.rotation.z = 0.0

                            time_found = rospy.Time(0)
                            self.pub_found_aruco.publish(time_found)
                            self.tfbr.sendTransform(msg_out)
            return frame

			
    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time(0)
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

        self.aruco_pub.publish(msg_out)

        msg = Bool()
        msg.data = self.landing
        
        self.land_pub.publish(msg)


def main():
    rospy.init_node('EGH450_vision', anonymous=True)
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()

    rospy.spin()
