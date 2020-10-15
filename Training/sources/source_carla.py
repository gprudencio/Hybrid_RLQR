from sources.source import source
from sources.carla.env import *
import signal
import sys
import cv2
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


##### SOURCE CARLA
class source_carla( source ):

    # Discrete actions:
        # 0 - Throttle
        # 1 - Throttle and right steer
        # 2 - Throttle and left steer
        # 3 - Brake
        # 4 - None

    ### __INIT__
    def __init__( self ):

        source.__init__( self )

        class Args:

            debug = True
            host = '127.0.0.1'
            port = 2000
            autopilot = False
            res = '500x300'
            width, height = [int(x) for x in res.split('x')]
            number_of_vehicles = 0
            number_of_pedestrians = 0
            desired_speed = 30
            continuous = True

        self.args = Args()
        self.env = CarlaEnv()

        # Open Server
        self.env.open_server(self.args)
        # Open Client
        self.env.init(self.args)

        def signal_handler(signal, frame):
            print('\nProgram closed!')
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

    ### INFORMATION
    def num_actions_lon( self ):
        return 2
    def num_actions_lat( self ):
        return 1
    def range_actions( self ):
        return 1

    ### START SIMULATION
    def start( self ):

        obsv, rewd, done = self.env.step([0,0,0,0,0,0,0,0])

        image     = self.process(obsv[0])
        matrix    = obsv[1]
        collision = obsv[2]
        speed     = obsv[3]
        distance_to_goal = obsv[4]
        emotional_states = obsv[5]

        info = [speed]

        return image, info

    ### MOVE ONE STEP
    def move( self , actn ):

        obsv, rewd, done = self.env.step(actn)

        image     = self.process(obsv[0])
        matrix    = obsv[1]
        collision = obsv[2]
        speed     = obsv[3]
        distance_to_goal = obsv[4]
        emotional_states = obsv[5]

        info = [speed]

        return image, rewd, done, info
    
    # Reference to control
    def ref( self ):
        
        tr, st = self.env.reference()
        
        return tr, st

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        # Convert image to gray
        obsv = np.uint8(obsv)
        #obsv = cv2.resize( obsv , ( 200 , 200 ) )
        
        obsv = cv2.resize( obsv , ( 168 , 84 ) ) # Deu certo uma vez
        
#        obsv = cv2.cvtColor( obsv , cv2.COLOR_BGR2GRAY )
#        _ , obsv = cv2.threshold( obsv , 100 , 255 , cv2.THRESH_BINARY )

#        # Plot the ANN image input:
#        fig = plt.figure()
#        for i in range( 1 ):
#            plt.subplot( 2 , 1 , i + 1 )
#            plt.imshow( obsv[:,:] , cmap = 'gray' )
#        plt.savefig('./auxiliar/rgb.png')
#        plt.close()

        return obsv
