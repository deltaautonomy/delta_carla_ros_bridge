# This script mimics a RADAR sensor. It request all its transformations from the ClientRADAR script.
# Refer that script for all transformations and bounding box values of all vehicles in world or ego
# vehicle frame

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from radar_utils import *


class RADAR:
    ''' This class instantiates a RADAR object'''

    def __init__(self, theta, r):
        ''' theta is the angular range of the RADAR.
        range is the distance up to which RADAR detects objects'''
        self.theta = theta
        self.r = r


class Vehicle:
    ''' instantiates a vehicle object. It has x,y of centroid and
    bounding box with points (x_max, y_max) and (x_min,y_min)'''

    # NOTE: y_max is not actually max y co-ordinate and similarly for y_min
    # NOTE: Get actual vehicle ID to add as an attribute
    def __init__(self, obj_id, x, y, z, x_max, y_max, z_max, x_min, y_min, z_min, velocity):
        self.id = obj_id
        self.x = x
        self.y = y
        self.z = z
        self.x_max = x_max  # box looks like [[x_max,y], [x_min,y]]
        self.y_max = y_max
        self.z_max = z_max
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.velocity = velocity


def field_of_view_filter(env_vehicles, radar):
    '''Given a list of vehicles in the environment. This function filters all
    those vehicles that are not in the field of view(FOV) of the RADAR.
    It returns a list of vehicles (x,y,yaw) of all vehicles in FOV'''
    # Iterate over all y values of vehicles
    filtered_vehicles = []
    for vehicle in env_vehicles:
        # Take the left most point of the vehicle and take abs
        x = vehicle.x_min
        y = vehicle.y_min
        # print(vehicle.x, vehicle.y)
        if x >= 1 / np.tan(radar.theta / 2) * abs(y) and x <= radar.r and x > 0:
            filtered_vehicles.append(vehicle)
    return filtered_vehicles

# Helper function to add RADAR based on some function
def noiseFunction(x_pos, RADAR, model):
    ''' Returns RADAR noise based on distance of object from RADAR'''
    scaling_factor = RADAR.r
    # Linear function
    if model == "linear":
        temp = 3*x_pos # std_dev cannot be negative
        noise = np.random.normal(0, np.abs(temp))/scaling_factor
        return noise
    # Quadratic Function
    if model == "quadratic":
        temp = 2*x_pos*x_pos # std_dev cannot be negative
        noise = np.random.normal(0, np.abs(temp))/(scaling_factor*scaling_factor)
        return noise

def radar_noise(std_dev):
    # Assume n is in meters. n=1 means that the Radar has std dev of 1 meter.
    return np.random.normal(0, std_dev)


def add_radar_noise(detected_vehicles, RADAR, model):
    for vehicle in detected_vehicles:
        noise = noiseFunction(vehicle.x, RADAR, "linear")
        vehicle.y_max += noise/3
        vehicle.y_min += noise/3
        vehicle.y += noise/3
        vehicle.x_max += noise
        vehicle.x_min += noise
        vehicle.x += noise

        vehicle.velocity[0] = vehicle.velocity[0] + noise/2
        vehicle.velocity[1] = vehicle.velocity[0] + noise
    return detected_vehicles




def radar_detect_vehicles(env_vehicles, radar):
    '''Returns list of vehicle objects detected by RADAR'''

    fov_vehicles = field_of_view_filter(env_vehicles, radar)
    if len(fov_vehicles) == 0: return []
    
    # Sort vehicles based on x values
    fov_vehicles = sorted(fov_vehicles, key=lambda x: x.x)
    
    # For every detected vehicle my RADAR will not be able to detect the
    # vehicles other detected vehicles are blocking. Every equation will have a slope
    slopes = []
    detected_vehicles = []
    flag = False
    
    # Iterate through FOV vehicles already sorted based on x
    for vehicle in fov_vehicles:
        x_max = vehicle.x_max
        x_min = vehicle.x_min
        y_max = vehicle.y_max
        y_min = vehicle.y_min
        
        # Find slopes of the detected vehicle
        m1 = np.tan(x_max / y_max)
        m2 = np.tan(x_min / y_min)
        if vehicle == fov_vehicles[0]:
            slopes.append([m1, m2])
            detected_vehicles.append(vehicle)
        else:
            for m in slopes:
                if m1 < m[0] or m2 > m[1]:
                    flag = True
                else:
                    flag = False
        if flag == True:
            slopes.append([m1, m2])
            detected_vehicles.append(vehicle)
    
    # Adding noise
    model = "linear"
    detected_vehicles = add_radar_noise(detected_vehicles, radar, model)
    return detected_vehicles


def parse_velocity(ego_vehicle, vehicle):
    '''Returns velocity with some added noise'''
    H_W_to_ego = get_car_bbox_transform(ego_vehicle)
    x_vel = vehicle.get_velocity().x
    y_vel = -vehicle.get_velocity().y
    z_vel = vehicle.get_velocity().z
    # The above values are in the world coordinate frame
    # Transforming them w.r.t ego vehicle frame
    vel_vec = np.array([x_vel, y_vel, z_vel, 0])
    vel_ego = np.matmul(np.linalg.pinv(H_W_to_ego), vel_vec)
    vel_ego = vel_ego.tolist()

    return vel_ego


# This is the final function that needs to run in order to visualize
# a RADAR output
def simulate_radar(theta, r, actor_list, ego_vehicle, radar_transform):
    ''' Simulate and visualize RADAR output'''
    radar = RADAR(theta, r)
    env_vehicles = []
    # Iterate over vehicles in actor list and create vehicle class objects
    for obj in actor_list:
        ego_velocity = parse_velocity(ego_vehicle, obj)
        x, y, z, x_max, y_max, z_max, x_min, y_min, z_min = get_min_max_bbox(ego_vehicle, obj, radar_transform)
        # All these values are w.r.t the center of the ego vehicle.
        # Tranform these values to the RADAR position
        pose = np.array([[x, y, z, 1], [x_max, y_max, z_max, 1], [x_min, y_min, z_min, 1]])
        vehicle = Vehicle(obj.id, x, y, z, x_max, y_max, z_max, x_min, y_min, z_min, ego_velocity)
        env_vehicles.append(vehicle)

    detected_vehicles = radar_detect_vehicles(env_vehicles, radar)    
    return detected_vehicles

def simulate_radar_GT(theta, r, actor_list, ego_vehicle, radar_transform):
    ''' Simulate and visualize RADAR output'''
    radar = RADAR(theta, r)
    env_vehicles = []
    # Iterate over vehicles in actor list and create vehicle class objects
    for obj in actor_list:
        ego_velocity = parse_velocity(ego_vehicle, obj)
        x, y, z, x_max, y_max, z_max, x_min, y_min, z_min = get_min_max_bbox(ego_vehicle, obj, radar_transform)
        # All these values are w.r.t the center of the ego vehicle.
        # Tranform these values to the RADAR position
        pose = np.array([[x, y, z, 1], [x_max, y_max, z_max, 1], [x_min, y_min, z_min, 1]])
        vehicle = Vehicle(obj.id, x, y, z, x_max, y_max, z_max, x_min, y_min, z_min, ego_velocity)
        env_vehicles.append(vehicle)
    return detected_vehicles

def visualize_radar(env_vehicles, detected_vehicles, radar, Truck):
    '''This function shows a scatter plot of vehicles in environment,
    the RADAR detected vehicles, and the ego vehicle'''
    env_plot = [[env_v.x, env_v.y] for env_v in env_vehicles]
    env_plot = np.asarray(env_plot)

    det_plot = [[det_v.x, det_v.y] for det_v in detected_vehicles]
    det_plot = np.asarray(det_plot)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(env_plot[:, 0], env_plot[:, 1], s=10, c='b', marker="s", label='All vehicles')
    if len(detected_vehicles) != 0:
        ax1.scatter(det_plot[:, 0], det_plot[:, 1], s=10, c='r', marker="o", label='Detected Vehicles')
    ax1.scatter(Truck.x, Truck.y, s=15, c='g', marker="+", label='Ego Vehicle')
    
    # Plot RADAR FOV
    xlim = np.tan(radar.theta / 2) * radar.r
    x = np.linspace(-xlim, xlim, 100)
    y = 1 / np.tan(radar.theta / 2) * abs(x)
    ax1.plot(x, y, '-r', label='FOV')
    
    # Set limits for plot
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-500, 500)
    plt.legend(loc='upper left')
    plt.show()
