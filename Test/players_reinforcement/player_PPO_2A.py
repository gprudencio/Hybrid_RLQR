from players_reinforcement.player_PPO_2 import *


# PLAYER PPO
class player_PPO_2A( player_PPO_2 ):

   
    NUM_FRAMES     = 3
    LEARNING_RATE  = 5e-4
    UPDATE_SIZE    = 5
    BATCH_SIZE     = 1024
    BATCH_SIZE_MLP = 64 #512
    EPSILON        = 0.12
    GAMMA          = 0.99
    LAM            = 0.95
    rgb            = 3 # black and white + v
    INFO_LEN       = 1 # info = speed,es1
    
    ### __INIT__
    def __init__( self ):

        player_PPO_2.__init__( self )
       
      
    # PROCESS OBSERVATION
    def process(self, obsv):

        obsv = np.stack( tuple( self.obsv_list[i] for i in range( self.NUM_FRAMES ) ), axis = -1 )


        if self.rgb > 1: obsv = obsv.reshape(-1, self.obsv_shape[0], self.obsv_shape[1], self.NUM_FRAMES * self.rgb)[0]

        return obsv

    ### PREPARE NETWORK
    def network( self ):
        
       
        # Actor

        Actor = self.brain.addBlock( 'Actor' )        
        
        Actor.addInput( shape = [ None, self.obsv_shape[0], self.obsv_shape[1], self.NUM_FRAMES * self.rgb ],
                        name  = 'Observation' )

        Actor.setLayerDefaults( type          = tb.layers.conv2d,
                                 activation    = tb.activs.relu,
                                 pooling       = 2,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        Actor.addLayer( out_channels = 64,   ksize = 8, strides = 4, input = 'Observation', name = 'Cnn1' )
        Actor.addLayer( out_channels = 96,   ksize = 4, strides = 2, name = 'Cnn2' )
        Actor.addLayer( out_channels = 128,  ksize = 3, strides = 1, name = 'Cnn3'  )

        Actor.setLayerDefaults( type       = tb.layers.fully,
                                activation = tb.activs.relu )

#        Actor.addLayer( out_channels = 1024, input = 'Cnn3', name = 'Hidden')
        Actor.addLayer( out_channels = 512, input = 'Cnn3', name = 'Hx')
        Actor.addLayer( out_channels = 1024, input = 'Hx', name = 'Hidden')

        Actor.addLayer( out_channels = self.num_actions_lat , input = 'Hidden', activation = None, name = 'Mu')
        Actor.addLayer( out_channels = self.num_actions_lat , input = 'Hidden', activation = tb.activs.softplus, name = 'Sigma', activation_pars = 0.5 )
        Actor.addLayer( out_channels = self.num_actions_lat , input = 'Hidden', activation = tb.activs.softmax,  name = 'Discrete' )

        mu     = Actor.tensor( 'Mu' )
        sigma  = Actor.tensor( 'Sigma' )
        dist   = tb.extras.dist_normal( mu, sigma )
        action = dist.sample( 1 )
        Actor.addInput( tensor = action, name = 'Output')

        # OldActor

        Old = self.brain.addBlock( 'Old' )

        Old.addInput( shape = [ None, self.obsv_shape[0], self.obsv_shape[1], self.NUM_FRAMES * self.rgb ],
                      name  = 'Observation' )

        Old.setLayerDefaults( type          = tb.layers.conv2d,
                                 activation    = tb.activs.relu,
                                 pooling       = 2,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        Old.addLayer( out_channels = 64,   ksize = 8, strides = 4, input = 'Observation' )
        Old.addLayer( out_channels = 96,   ksize = 4, strides = 2 )
        Old.addLayer( out_channels = 128,  ksize = 3, strides = 1, name = 'Cnn'  )

        Old.setLayerDefaults( type       = tb.layers.fully,
                              activation = tb.activs.relu )

#        Old.addLayer( out_channels = 1024, input = 'Cnn', name = 'Hidden')
        Old.addLayer( out_channels = 512, input = 'Cnn', name = 'Hx')
        Old.addLayer( out_channels = 1024, input = 'Hx', name = 'Hidden')

        Old.addLayer( out_channels = self.num_actions_lat , input = 'Hidden', activation = None, name = 'Mu')
        Old.addLayer( out_channels = self.num_actions_lat , input = 'Hidden', activation = tb.activs.softplus, name = 'Sigma', activation_pars = 0.5 )
        Old.addLayer( out_channels = self.num_actions_lat , input = 'Hidden', activation = tb.activs.softmax,  name = 'Discrete' )

        # Critic

        Critic = self.brain.addBlock( 'Critic' )

        Critic.addInput( shape = [ None, self.obsv_shape[0], self.obsv_shape[1], self.NUM_FRAMES * self.rgb ],
                         name  = 'Observation' )

        Critic.setLayerDefaults( type          = tb.layers.conv2d,
                                 activation    = tb.activs.relu,
                                 pooling       = 2,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        Critic.addLayer( weight_share = '../Actor/Cnn1', out_channels = 64,   ksize = 8, strides = 4, input = 'Observation' )
        Critic.addLayer( weight_share = '../Actor/Cnn2', out_channels = 96,   ksize = 4, strides = 2 )
        Critic.addLayer( weight_share = '../Actor/Cnn3', out_channels = 128,  ksize = 3, strides = 1, name = 'Cnn'  )

        Critic.setLayerDefaults( type       = tb.layers.fully,
                                 activation = tb.activs.relu )

#        Critic.addLayer( out_channels = 512, input = 'Cnn', name = 'Hidden')
        Critic.addLayer( out_channels = 512, input = 'Cnn', name = 'Hx')
        Critic.addLayer( out_channels = 1024, input = 'Hx', name = 'Hidden')

        Critic.addLayer( out_channels = 1, input = 'Hidden', name = 'Value', activation = None )


        # MLP_Actor
        MLP_Actor = self.brain.addBlock( 'MLP_A' )
        MLP_Actor.addInput( shape = [ None, self.INFO_LEN ], name = 'Info' )

        MLP_Actor.setLayerDefaults( type       = tb.layers.fully,
                                    activation = tb.activs.relu,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        MLP_Actor.addLayer( out_channels = 64, input = 'Info', name = 'layer_1' )
        MLP_Actor.addLayer( out_channels = 64, name = 'layer_2' ) 
        MLP_Actor.addLayer( out_channels = 64, name = 'layer_3' )
        
        MLP_Actor.addLayer( out_channels = self.num_actions_lon, input = 'layer_3', activation = tb.activs.softmax, name = 'MLP_Discrete' )
        MLP_Actor.addLayer( out_channels = self.num_actions_lon, input = 'layer_3', activation = tb.activs.softmax, name = 'MLP_Mu')
        MLP_Actor.addLayer( out_channels = self.num_actions_lon, input = 'layer_3', activation = tb.activs.softplus, name = 'MLP_Sigma', activation_pars = 0.5 )

        mu     = MLP_Actor.tensor( 'MLP_Mu' )
        sigma  = MLP_Actor.tensor( 'MLP_Sigma' )
        dist   = tb.extras.dist_normal( mu, sigma )
        action = dist.sample( 1 )
        MLP_Actor.addInput( tensor = action, name = 'Output')

        # MLP_OldActor
        MLP_Old = self.brain.addBlock( 'MLP_O' )
        MLP_Old.addInput( shape = [ None, self.INFO_LEN ], name = 'Info' )

        MLP_Old.setLayerDefaults( type = tb.layers.fully,
                              activation = tb.activs.relu,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        MLP_Old.addLayer( out_channels = 64, input = 'Info', name = 'layer_1' )
        MLP_Old.addLayer( out_channels = 64, name = 'layer_2' ) 
        MLP_Old.addLayer( out_channels = 64, name = 'layer_3' )
        
        MLP_Old.addLayer( out_channels = self.num_actions_lon, input = 'layer_3', activation = tb.activs.softmax, name = 'MLP_Discrete' )
        MLP_Old.addLayer( out_channels = self.num_actions_lon, input = 'layer_3', activation = tb.activs.softmax, name = 'MLP_Mu')
        MLP_Old.addLayer( out_channels = self.num_actions_lon, input = 'layer_3', activation = tb.activs.softplus, name = 'MLP_Sigma', activation_pars = 0.5 )

        # MLP_Critic
        MLP_Critic = self.brain.addBlock( 'MLP_C' )
        MLP_Critic.addInput( shape = [ None, self.INFO_LEN ], name = 'Info' )

        MLP_Critic.setLayerDefaults( type       = tb.layers.fully,
                                     activation = tb.activs.relu,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        MLP_Critic.addLayer( out_channels = 64, input = 'Info' )
        MLP_Critic.addLayer( out_channels = 1, name = 'Value', activation = None )
