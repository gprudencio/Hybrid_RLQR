from collections import deque

from players_reinforcement.player import player
from auxiliar.aux_plot import *

import sys
sys.path.append('..')

import tensorblock as tb
import numpy as np
import tensorflow as tf

# PLAYER PPO
class player_PPO_2(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.num_stored_obsv = self.NUM_FRAMES
        self.experiences = []
        self.experiences_MLP = []

    # CHOOSE NEXT ACTION
    def act(self, state, info, tr, st):

        return self.calculate(state, info, tr, st)    
    
    # Lateral PID control
    def lat_control(self, x1, x2):      
        
        
        #K = [[1.11, 0.86]]  # K: DRL + LQR
        #K = [[1.04, 1.13]]  # K: DRL + KF + RLQR
        K = [[0.89, 0.90]] # KR: DRL + EAND + RLQR
                
        x = [[x1], [x2]]         
        a_lat = np.matmul(K,x) 
        
     
        return a_lat[0][0]
    ''' -------------------------------------------------- '''

    # CALCULATE NETWORK
    def calculate(self, state, info, tr, st):

        # Long action (3)
        lon_output = self.brain.run( 'MLP_A/Output', [ [ 'MLP_A/Info', [info] ] ] )
        lon_action = np.reshape( lon_output, self.num_actions_lon )

        # Lat action (3)
        lat_output = self.brain.run( 'Actor/Output', [ [ 'Actor/Observation', [state] ] ] )
        lat_action = np.reshape( lat_output, self.num_actions_lat )
        
        ''' CNN output '''
        lat_out = self.brain.run( 'Actor/Cnn3', [ [ 'Actor/Observation', [state] ] ] ) 
        state_variables = np.concatenate((lat_out, st, tr), axis=None)  
        
        ''' Error value '''         

        x528  = state_variables[528]
        x567  = state_variables[567]

        ref_528 = 0.752061230419881
        ref_567 = 0.842550496986767

        c_528 = 71.9093550000000
        c_567 = 45.9564550000000        
      
        
        e_528 = ref_528 - x528/c_528
        e_567 = x567/c_567 - ref_567
        
        x1 = e_528
        x2 = e_567        
        
        ''' Control LQR/RLQR ''' 
        a_lat = self.lat_control(x1, x2)         
        lat_action[0] = a_lat 
        ''' '''                 

        action =  np.concatenate((lon_action,lat_action))
        return action

    # PREPARE NETWORK
    def operations(self):


        # Placeholders

        self.brain.addInput( shape = [ None, self.num_actions_lat ], name = 'Actions_Lat'  )
        self.brain.addInput( shape = [ None, self.num_actions_lon ], name = 'Actions_Lon'  )      
        self.brain.addInput( shape = [ None, self.num_actions_lat ], name = 'O_Mu_Lat'  )
        self.brain.addInput( shape = [ None, self.num_actions_lon ], name = 'O_Mu_Lon'  )
        self.brain.addInput( shape = [ None, self.num_actions_lat ], name = 'O_Sigma_Lat'  )
        self.brain.addInput( shape = [ None, self.num_actions_lon ], name = 'O_Sigma_Lon'  )
        self.brain.addInput( shape = [ None, 1 ] ,                   name = 'Advantage' )

        ''' Operations '''
        
        ''' CNN '''

        ''' Critic '''
        self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                 input    = [ 'Critic/Value','Advantage' ],
                                 name     = 'CriticCost' )
        
        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'CriticCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'CriticOptimizer' )
           
        ''' Actor '''
        self.brain.addOperation( function = tb.ops.ppocost_distrib,
                                 input    = [ 'Actor/Mu',
                                              'Actor/Sigma',
                                              'O_Mu_Lat',
                                              'O_Sigma_Lat',
                                              'Actions_Lat',
                                              'Advantage',
                                              self.EPSILON ],
                                 name     = 'ActorCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'ActorCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'ActorOptimizer' )

        ''' Assign Old Actor '''
        
        self.brain.addOperation( function = tb.ops.assign,
                                 input = ['Old', 'Actor'],
                                 name = 'Assign' )

        ''' MLP ''' 

        ''' MLP_Critic '''
        self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                 input    = [ 'MLP_C/Value','Advantage' ],
                                 name     = 'MLP_CriticCost' )
        
        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'MLP_CriticCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'MLP_CriticOptimizer' )
        
        ''' MLP_Actor '''
        self.brain.addOperation( function = tb.ops.ppocost_distrib,
                                 input    = [ 'MLP_A/MLP_Mu',
                                              'MLP_A/MLP_Sigma',
                                              'O_Mu_Lon',
                                              'O_Sigma_Lon',
                                              'Actions_Lon',
                                              'Advantage',
                                              self.EPSILON ],
                                 name     = 'MLP_ActorCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'MLP_ActorCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'MLP_ActorOptimizer' )

            
        ''' MLP_Assign Old Actor '''
        self.brain.addOperation( function = tb.ops.assign,
                                 input = ['MLP_O', 'MLP_A'],
                                 name = 'MLP_Assign' )      


    # TRAIN NETWORK
    def train( self, prev_state, curr_state, actn, rewd, done, episode, prev_info, curr_info ):

        # Store New Experience Until Train
        self.experiences.append( (prev_state, curr_state, actn, rewd, done, prev_info, curr_info) )
        self.experiences_MLP.append( (prev_state, curr_state, actn, rewd, done, prev_info, curr_info) )

        # Check for Train
        if ( len(self.experiences) >= self.BATCH_SIZE ):

            batch = self.experiences

            # Separate Batch Data
            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions     = [d[2] for d in batch]
            rewards     = [d[3] for d in batch]
            dones       = [d[4] for d in batch]
            prev_info   = [d[5] for d in batch]
            curr_info   = [d[6] for d in batch]

            # Actions and Rewards Lat and Long
            actions_lat = np.expand_dims(np.array(actions)[:,2],1)
            rewards_lat = np.array(rewards)[:,1:]
            rewards_lat = np.sum(rewards_lat,axis=1)

            # States Values
            prev_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', prev_states  ] ] ) )
            curr_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', curr_states  ] ] ) )

            running_add_y = 0
            running_add_a = 0
            y = np.zeros_like(rewards_lat)
            advantage_lat = rewards_lat + (self.GAMMA * curr_values) - prev_values
            for t in reversed ( range( 0, len( advantage_lat ) ) ):
                if dones[t]:
                    curr_values[t] = 0
                    running_add_a  = 0
                running_add_y  = curr_values[t] * self.GAMMA + rewards_lat   [t]
                running_add_a  = running_add_a  * self.GAMMA * self.LAM + advantage_lat [t]
                y [t] = running_add_y
                advantage_lat [t] = running_add_a
            y = np.expand_dims( y, 1 )
            advantage_lat = np.expand_dims( advantage_lat, 1 )

            # Assign Old Pi
            self.brain.run( ['Assign'], [] )

            # Get Current Probabilities
            # a_Mu, a_Sigma = self.brain.run( [ 'Actor/Mu', 'Actor/Sigma'  ], [ [ 'Actor/Observation', curr_states ] ] )
            # Get Old Probabilities
            o_Mu, o_Sigma = self.brain.run( [ 'Old/Mu', 'Old/Sigma' ], [ [ 'Old/Observation', prev_states ] ] )
                        
                        
            for _ in range (self.UPDATE_SIZE):

                self.brain.run( [ 'ActorOptimizer' ], [ [ 'Actor/Observation', prev_states   ],
                                                        [ 'O_Mu_Lat',          o_Mu          ],
                                                        [ 'O_Sigma_Lat',       o_Sigma       ],
                                                        [ 'Actions_Lat',       actions_lat   ],
                                                        [ 'Advantage',         advantage_lat ] ], debug = True )


                self.brain.run( [ 'CriticOptimizer' ], [ [ 'Critic/Observation', prev_states ],
                                                         [ 'Advantage',          y           ] ] ) 
            # Reset
            self.experiences = []


#        # Check for Train
#        if ( len(self.experiences_MLP) >= self.BATCH_SIZE_MLP ):
#
#            batch = self.experiences_MLP
#
#            # Separate Batch Data
#            prev_states = [d[0] for d in batch]
#            curr_states = [d[1] for d in batch]
#            actions     = [d[2] for d in batch]
#            rewards     = [d[3] for d in batch]
#            dones       = [d[4] for d in batch]
#            prev_info   = [d[5] for d in batch]
#            curr_info   = [d[5] for d in batch]
#
#            actions_long = np.array(actions)[:,0:2]
#            rewards_long = np.array(rewards)[:,0]
#
#            MLP_prev_values = np.squeeze( self.brain.run( 'MLP_C/Value' , [ [ 'MLP_C/Info', prev_info  ] ] ) )
#            MLP_curr_values = np.squeeze( self.brain.run( 'MLP_C/Value' , [ [ 'MLP_C/Info', curr_info  ] ] ) )
#
#            running_add_y = 0
#            running_add_a = 0
#            MLP_y = np.zeros_like(rewards_long)
#            advantage_long = rewards_long + (self.GAMMA * MLP_curr_values) - MLP_prev_values
#            for t in reversed ( range( 0, len( advantage_long ) ) ):
#                if dones[t]:
#                    MLP_curr_values[t] = 0
#                    running_add_a  = 0
#                running_add_y  = MLP_curr_values[t] * self.GAMMA + rewards_long[t]
#                running_add_a  = running_add_a  * self.GAMMA * self.LAM + advantage_long [t]
#                MLP_y [t] = running_add_y
#                advantage_long [t] = running_add_a
#            MLP_y = np.expand_dims( MLP_y, 1 )
#            advantage_long = np.expand_dims( advantage_long, 1 )
#
#            self.brain.run( ['MLP_Assign'], [] )
#
#            # Get Old Probabilities
#            o_Mu, o_Sigma = self.brain.run( [ 'MLP_O/MLP_Mu', 'MLP_O/MLP_Sigma' ], [ [ 'MLP_O/Info', prev_info ] ] )
#
#            # Optimize
#            for _ in range (self.UPDATE_SIZE):
#                self.brain.run( [ 'MLP_ActorOptimizer' ], [ [ 'MLP_A/Info',  prev_info      ],
#                                                        [ 'O_Mu_Lon',        o_Mu           ],
#                                                        [ 'O_Sigma_Lon',     o_Sigma        ],
#                                                        [ 'Actions_Lon',     actions_long   ],
#                                                        [ 'Advantage',       advantage_long ] ] )
#                self.brain.run( [ 'MLP_CriticOptimizer' ], [ [ 'MLP_C/Info', prev_info      ],
#                                                             [ 'Advantage',  MLP_y          ] ] )
#
#            # Reset
#            self.experiences_MLP = []
