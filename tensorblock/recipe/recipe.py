
import tensorflow as tf
import tensorblock as tb

import numpy as np

class recipe( tb.recipe.base , tb.recipe.block , tb.recipe.init ,
              tb.recipe.input , tb.recipe.layer , tb.recipe.operation ,
              tb.recipe.plot , tb.recipe.print , tb.recipe.save ,
              tb.recipe.summary , tb.recipe.train ):

####### Initialize
    def __init__( self , sess = None , prev = None , name = None ):

        self.sess = tf.Session() if sess is None else sess
        self.root = self         if prev is None else prev.root

        self.folder = '' if name is None else name + '/'
        if prev is not None: self.folder = prev.folder + self.folder

        self.initDefaults()
        self.initVariables()

        self.strname = name

    def __str__( self ):

        return self.strname

####### Add Block
    def addBlock( self , name = None ):

        name = self.add_label(
                self.blocks , 'Block' , name , add_order = True )

        self.blocks.append( [ tb.recipe( sess = self.sess , prev = self , name = name ) , name ] )
        return self.tensor( name )

####### Eval
    def eval( self , names , dict = None, debug=False ):        
        
#        if debug:
#            
#            query_vars_X    = [ 'ActorCost' ] 
#            tensors_X = self.tensor_list(query_vars_X )
#            outputs_X = self.sess.run( tensors_X , feed_dict = dict )
#                        
#            with open('Cost.txt', 'a') as f:  
#                f.write('%f ' %outputs_X[0])
#                f.write('\n')   
                       

        tensors = self.tensor_list( names )
        if not isinstance( names , list ) and len( tensors ) == 1 : tensors = tensors[0]
        
        outputs = self.sess.run( tensors , feed_dict = dict )

        return outputs

####### Run
    def run( self , names , inputs , use_dropout = True, debug=False ):

        dict = {}

        inputs = tb.aux.parse_pairs( inputs )
        for data in inputs:
            dict[ self.node( data[0] ) ] = data[1]

        for i in range( len( self.root.dropouts ) ):
            if use_dropout: dict[ self.root.dropouts[i][0] ] = self.root.dropouts[i][1][1]
            else:           dict[ self.root.dropouts[i][0] ] = 1.0
            
        outputs = self.eval( names , dict, debug=debug )

        return outputs

####### Assign
    def assign( self , names , values ):

        if not isinstance( values , list ): values = [ values ]
        tensors = self.tensor_list( names )

        for i , tensor in enumerate( tensors ):
            tensor.assign( values[i] ).eval( session = self.sess )
