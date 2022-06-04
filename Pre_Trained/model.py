'''
SS-L8: HxWxC = 256x1024x3
'''

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.keras import layers

def Upsampling(inputs,scale):
    return layers.UpSampling2D(size=(scale, scale))(inputs)

def build(inputs, num_classes):  

    ### Initial stage
    skip0 = inputs                                           
    net = slim.conv2d(inputs, 32, [2,2], padding="SAME")     
    net = tf.nn.relu(net)
  
    ### Down-Sampling Path
    ## Encoding Block 1
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip1 = net     
    net = slim.conv2d(net, 48, [2,2], padding="SAME")              
    net = tf.nn.relu(net)  
    
    ## Encoding Block 2
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip2 = net   
    net = slim.conv2d(net, 96, [2,2], padding="SAME")               
    net = tf.nn.relu(net)

    ## Encoding Block 3
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip3 = net    
    net = slim.conv2d(net, 128, [2,2], padding="SAME")               
    net = tf.nn.relu(net)    
    
    ## Encloding Block 4
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip4 = net    
    net = slim.conv2d(net, 192, [2,2], padding="SAME")               
    net = tf.nn.relu(net)      
    
    ## Encloding Block 5
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip5 = net    
    net = slim.conv2d(net, 256, [2,2], padding="SAME")               
    net = tf.nn.relu(net) 
    
    ## Encloding Block 6
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip6 = net    
    net = slim.conv2d(net, 320, [2,2], padding="SAME")               
    net = tf.nn.relu(net) 

    ## Encloding Block 7
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip7 = net    
    net = slim.conv2d(net, 384, [2,2], padding="SAME")              
    net = tf.nn.relu(net)
    
    ## Encloding Block 8
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  

   
    ### Upsampling Path 
    ## Decoding Block 1
    net = Upsampling(net, 2)                             
    net = tf.concat([net, skip7], axis=-1)              
    net = slim.conv2d(net, 384, [2,2], padding="SAME")   
    net = tf.nn.relu(net)
    
    ## Decoding Block 2
    net = Upsampling(net, 2)                             
    net = tf.concat([net, skip6], axis=-1)               
    net = slim.conv2d(net, 320, [2,2], padding="SAME")   
    net = tf.nn.relu(net)

    ## Decoding Block 3
    net = Upsampling(net, 2)                             
    net = tf.concat([net, skip5], axis=-1)               
    net = slim.conv2d(net, 256, [2,2], padding="SAME")   
    net = tf.nn.relu(net)

    ## Decoding Block 4
    net = Upsampling(net, 2)                             
    net = tf.concat([net, skip4], axis=-1)               
    net = slim.conv2d(net, 192, [2,2], padding="SAME")   
    net = tf.nn.relu(net)

    ## Decoding Block 5
    net = Upsampling(net, 2)                             
    net = tf.concat([net, skip3], axis=-1)               
    net = slim.conv2d(net, 128, [2,2], padding="SAME")   
    net = tf.nn.relu(net)

    ## Decoding Block 6
    net = Upsampling(net, 2)                             
    net = tf.concat([net, skip2], axis=-1)               
    net = slim.conv2d(net, 96, [2,2], padding="SAME")    
    net = tf.nn.relu(net)
    
    ## Decoding Block 7
    net = Upsampling(net, 2)                            
    net = tf.concat([net, skip1], axis=-1)               
    net = slim.conv2d(net, 48, [2,2], padding="SAME")    
    net = tf.nn.relu(net)

    ## Decoding Block 8
    net = Upsampling(net, 2)                            
    net = tf.concat([net, skip0], axis=-1)               
    net = slim.conv2d(net, 32, [2,2], padding="SAME")    
    net = tf.nn.relu(net)
    
    ### Softmax
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')  
    return net