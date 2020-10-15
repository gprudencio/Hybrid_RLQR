import numpy as np
import math

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Defines the observation matrix
def observation_matrix( actor_list, car_info, inter_x_left, inter_x_right ):
    '''
    Observation Matrix:
    5 informations for each placeholder: | class | distance x | distance y | vel. | rotation
    4 classes: 0:null, 1:street right intersection, 2:street left intersection, 3:dynamic objects
    '''

    # Car info
    car_id    = car_info[0]
    car_theta = car_info[1]
    car_x     = np.abs(car_info[2])
    car_y     = np.abs(car_info[3])
    inter_x_left  = np.abs(inter_x_left)
    inter_x_right = np.abs(inter_x_right)
    start_pos_x = np.abs(car_info[5])
    start_pos_y = np.abs(car_info[6])

    # Matrix info
    height      = 5.0
    width       = 10.0
    h_divisions = 50
    w_divisions = 50

    # Create matrix template to be filled
    matrix = np.zeros((w_divisions,h_divisions,5))
    for i in range(w_divisions):

        # Current matrix width
        current_matrix_w = (i) * width/w_divisions - width / 2

        # Placeholders segmentation in line
        for j in range(h_divisions):

            # Current matrix height
            current_matrix_h = height * ((j+1)/h_divisions)
            # Class
            matrix[i,j,0] = 0
            # Point x and y of each point of the matrix
            m_x, m_y = cartesian( current_matrix_w, current_matrix_h, car_theta, car_x, car_y, start_pos_x, start_pos_y )
            matrix[i,j,1] = m_x
            matrix[i,j,2] = m_y
            # Velocity of the other car
            matrix[i,j,3] = 0 # 0 to all but car
            # Rotation of the other car
            matrix[i,j,4] = 0 # 0 to all but car

    # Other vehicles
    other_vehicles = actor_list.filter('vehicle.*')
    other_vehicles = [v for v in other_vehicles if v.id != car_id]

    # Lights
    lights_list = actor_list.filter('traffic_light.*')

    # Speed sign
    speed_sign = actor_list.filter('traffic.speed_limit.*')

    # List of nearby objects
    nearby_list = []
    for agent in other_vehicles:

        other_velocity    = agent.get_velocity()
        othercar_speed    = math.sqrt(other_velocity.x**2 + other_velocity.y**2 + other_velocity.z**2)
        othercar_position = agent.get_location()
        othercar_rotation = agent.get_transform().rotation.yaw
        if agent_is_in_rectangle(othercar_position, car_x, car_y, height, width):
            nearby_list.append( ["car", 3, othercar_position.x, othercar_position.y, othercar_speed, othercar_rotation] )

    # Update matrix classes with right and left street intersections
    # Apply transformation coordinates
    for i in range(w_divisions):
        # Placeholders segmentation in line
        for j in range(h_divisions):
            if matrix[i,j,1] <  inter_x_left:
                matrix[i,j,0] = 2  # street left intersection
            if matrix[i,j,1] >  inter_x_right:
                matrix[i,j,0] = 1  # street right intersection

    # Update matrix classes with new agents info (dynamic)
    for agent in nearby_list:

        if (agent[0] == "car"):
            center_x = agent[2]
            center_y = agent[3]
            size_x, size_y  = 3.0, 2.0

            for i in range(w_divisions):
                # Placeholders segmentation in line
                for j in range(h_divisions):
                    matrix_x = matrix[i,j,1]
                    matrix_y = matrix[i,j,2]

                    # Check if car is in the position of each matrix position
                    if( np.abs(matrix_x - center_x ) < size_x/2 and np.abs(matrix_y - center_y ) < size_y/2 ):
                        # Then there is car in this position
                        # Class
                        matrix[i,j,0] = 3
                        # Velocity of the other car
                        matrix[i,j,3] = agent[4]
                        # Rotation of the other car
                        matrix[i,j,4] = agent[5]

    # Print matrix
    #print_matrix = matrix[:,:,0]
    #fig = plt.figure()
    #sns.heatmap(print_matrix, annot=False, square=True, vmin=0, vmax=3)
    #plt.savefig('./auxiliar/matrix.png')
    #plt.close()

    return matrix

# Auxiliar function to see if agent is rectangle
def agent_is_in_rectangle( agent_position, car_x, car_y, height, width ):
    if np.abs(np.abs(agent_position.x) - np.abs(car_x)) <  height:
        if np.abs(np.abs(agent_position.y) - np.abs(car_y)) < width:
            return True
    return False

# Auxiliar function to transform coordinates
def cartesian( w, h, theta, car_x, car_y, start_pos_x, start_pos_y ):

    rho, phi = cart2pol( h, w)
    x, y = pol2cart( rho, phi + np.deg2rad(theta) )
    x = x + car_x - start_pos_x
    y = y + car_y - start_pos_y

    return x, y

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
