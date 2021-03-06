 
import tensorflow as tf
import tensorblock as tb

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
        
        print('Bus EVAL')
        print(names)
        
        if debug:
            query_vars    = [ 'Actor/Mu',
                              'Actor/Sigma',
                              'O_Mu_Lat',
                              'O_Sigma_Lat',
                              'Actions_Lat',
                              'Advantage']
            
            tensors2 = self.tensor_list(query_vars )
            outputs2 = self.sess.run( tensors2 , feed_dict = dict )
            print(outputs2)            
            
            print('---------------------')
            print('Actor cost')
            query_vars_X    = [ 'ActorCost' ] 
            tensors_X = self.tensor_list(query_vars_X )
            outputs_X = self.sess.run( tensors_X , feed_dict = dict )
            print(outputs_X)  
        

        tensors = self.tensor_list( names )
        if not isinstance( names , list ) and len( tensors ) == 1 : 
            tensors = tensors[0]
            
#        print(dict)
        
#        pred = tensors.eval(feed_dict = dict)
#        print(pred)        
        
#        pred = sess.run(tf_pred, feed_dict={input_node: frame})
#        pred = tf_pred.eval(frame)
        
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        
        
#        print('eval out') 
        outputs = self.sess.run( tensors , feed_dict = dict )
#        print(outputs)

        return outputs

####### Run
    def run( self , names , inputs , use_dropout = True, debug=False ):
        
#        print('Bus in')

        dict = {}

        inputs = tb.aux.parse_pairs( inputs )
        
#        print(names)
#        print(inputs)
        
        for data in inputs:
#            print('Bus midle 1')
            dict[ self.node( data[0] ) ] = data[1]
            
#        print(dict)

        for i in range( len( self.root.dropouts ) ):
            print('Bus midle 2')
            print(i)
            if use_dropout:
                print('L1')
                dict[ self.root.dropouts[i][0] ] = self.root.dropouts[i][1][1]
            else:        
                print('L2')
                dict[ self.root.dropouts[i][0] ] = 1.0

#        print('Bus out')
        
        outputs = self.eval( names , dict, debug=debug )
        
#        print(outputs)

        return outputs

####### Assign
    def assign( self , names , values ):
        
        print('Bus Assign')

        if not isinstance( values , list ): values = [ values ]
        tensors = self.tensor_list( names )

        for i , tensor in enumerate( tensors ):
            tensor.assign( values[i] ).eval( session = self.sess )
