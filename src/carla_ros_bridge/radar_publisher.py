#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Prateek Parmeshwar
Email   : pparmesh@andrew.cmu.edu
Version : 1.0.0
Date    : Apr 08, 2019
'''

import random
import threading

import tf
import carla
import rospy
import numpy as np
from geometry_msgs.msg import Point, Polygon, Vector3
from radar_msgs.msg import RadarTrack, RadarTrackArray

from carla_ros_bridge.bridge import CarlaRosBridge
from carla_ros_bridge.bridge_with_rosbag import CarlaRosBridgeWithBag
from carla_ros_bridge.radar import simulate_radar
from delta_tracking_fusion.msg import Track, TrackArray

RADAR_FRAME = '/ego_vehicle/radar'
VEHICLE_FRAME = '/ego_vehicle'


def list_to_ros_vector3(input_list):
    output_vector = Vector3()
    output_vector.x = input_list[0]
    output_vector.y = input_list[1]
    output_vector.z = input_list[2]
    return output_vector


def polygon_list_to_ros_points(vehicle):
    output_polygon = Polygon()
    
    mid_point = Point()
    mid_point.x, mid_point.y, mid_point.z = vehicle.x, vehicle.y, vehicle.z
    output_polygon.points.append(mid_point)
    
    max_point = Point()
    max_point.x, max_point.y, max_point.z = vehicle.x_max, vehicle.y_max, vehicle.z_max
    output_polygon.points.append(max_point)
    
    min_point = Point()
    min_point.x, min_point.y, min_point.z = vehicle.x_min, vehicle.y_min, vehicle.z_min
    output_polygon.points.append(min_point)
    
    return output_polygon


def publisher(actor_list, ego_vehicle):
    # Setup node
    pub = rospy.Publisher('/carla/ego_vehicle/radar/tracks', RadarTrackArray, queue_size=10)
    # Change 1
    pub_ground_truth = rospy.Publisher('/carla/ego_vehicle/tracks/ground_truth', TrackArray, queue_size=10)

    # Define RADAR parameters
    theta_range = 2 * np.pi / 3
    dist_range = 150
    # x, y, z, roll, pitch, yaw
    radar_transform = [2.2, 0, 0.5, 0, 0, 0]
    # Chnage 2
    ground_truth_transform = [0, 0, 0, 0, 0, 0]

    # Publish at a rate of 13Hz. This is the RADAR frequency
    r = rospy.Rate(13)

    # Randomly publish some data
    while not rospy.is_shutdown():
        msg = RadarTrackArray()
        ground_truth_msg = TrackArray()
        br = tf.TransformBroadcaster()
        br.sendTransform((radar_transform[0], radar_transform[1], radar_transform[2]), \
                      tf.transformations.quaternion_from_euler(radar_transform[3], radar_transform[4], radar_transform[5]), \
                      rospy.Time.now(), RADAR_FRAME, VEHICLE_FRAME)

        # Get list of all detected vehicles
        radar_detections = simulate_radar(theta_range, dist_range,
            actor_list, ego_vehicle, radar_transform)
        # Change 3
        ground_truth_detections = simulate_radar_ground_truth(theta_range, dist_range,
            actor_list, ego_vehicle, ground_truth_transform)

        # Iterate over all cars and store their values in RADAR_msgs
        for detection in radar_detections:
            radar_track = RadarTrack()
            # Assigning RADAR ID
            radar_track.track_id = detection.id
            # Assingning three detected points
            radar_track.track_shape = polygon_list_to_ros_points(detection)
            # Assigning linear velocity
            radar_track.linear_velocity = list_to_ros_vector3(detection.velocity)
            # Append to main array
            msg.tracks.append(radar_track)

         # Change 4   
        for ground_truth_vehicle in ground_truth_detections:
            ground_truth_track = Track()
            # Assigning label
            ground_truth_track.label = "vehicle"
            # Assingning position
            ground_truth_track.x = ground_truth_vehicle.x
            ground_truth_track.y = ground_truth_vehicle.y
            # Assigning linear velocity
            ground_truth_track.vx = ground_truth_vehicle.velocity[0]
            ground_truth_track.vy = ground_truth_vehicle.velocity[1]
            # Assign covariance
            ground_truth_track.covariance = list(np.eye(4, dtype=np.float64).flatten())
            # Assign track ID
            ground_truth_track.track_id = ground_truth_vehicle.id
            # Append to main array
            ground_truth_msg.tracks.append(ground_truth_track)


        # Header stamp and publish the message
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = RADAR_FRAME
        pub.publish(msg)

        # Change 5
        ground_truth_msg.header.stamp = rospy.Time.now()
        ground_truth_msg.header.frame_id = VEHICLE_FRAME
        pub_ground_truth.publish(ground_truth_msg)
        r.sleep()


def main():
    """
    main function for carla simulator ROS bridge
    maintaiing the communication client and the CarlaRosBridge objects
    """
    rospy.init_node("radar_client", anonymous=True)

    params = rospy.get_param('carla')
    host = params['host']
    port = params['port']

    rospy.loginfo("Trying to connect to {host}:{port}".format(host=host, port=port))

    try:
        carla_client = carla.Client(host=host, port=port)
        carla_client.set_timeout(2000)

        carla_world = carla_client.get_world()

        rospy.loginfo("Connected")

        # bridge_cls = CarlaRosBridge
        # carla_ros_bridge = bridge_cls(
        #     carla_world=carla_client.get_world(), params=params)
        # carla_ros_bridge.run()
        # carla_ros_bridge = None
        # Get all  vehicle actors in environment
        actor_list = carla_world.get_actors().filter('vehicle.*')
        npc_list = []

        # Get ego vehicle object and remove it from actor list
        ego_vehicle = None
        for actor in actor_list:
            attribute_val = actor.attributes['role_name']
            if attribute_val == 'hero':
                ego_vehicle = actor
            else:
                npc_list.append(actor)

        if ego_vehicle is not None:
            publisher(npc_list, ego_vehicle)

        rospy.logdebug("Delete world and client")
        del carla_world
        del carla_client

    finally:
        rospy.loginfo("Done")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print('ROSInterruptException occurred')
