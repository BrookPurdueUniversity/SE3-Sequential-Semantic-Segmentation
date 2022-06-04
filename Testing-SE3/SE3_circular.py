
import os,time,cv2
import numpy as np
from glob import glob
import tensorflow as tf
tf.enable_eager_execution()
import easydict
import csv
from tensorflow.contrib import slim
from tensorflow.keras import layers

###############################################################################
###############################################################################
    
def conv(tensor, weights, biases): 
    conv = tf.nn.conv2d(tensor, weights, strides=[1,1,1,1], padding='SAME')
    output = conv + biases
    return output

###############################################################################
###############################################################################
    
def get_label_info(csv_path):
    """
    # Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!
    
    @csv_path: 
        The file path of the class dictionairy stored in a csv file
        
    @return: 
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")
    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values

def load_image(path):
    """
    @path:
        absolute path to an image
    
    @return=[H, W, C]:
        an image array
    """
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image
    
def reverse_one_hot(image):
    """
    # Transform an array in one-hot format (C=num_classes),
    to an array with only 1 channel (C=1), where each pixel value denotes the sementic class index.
    
    @image=[T,H,W,C=num_class]:
        The one-hot format image array.
        
    @return=[T,H,W]:
        An array with the same shape of the input image array, but
        with a depth size of 1 (C=1), where each pixel value denotes the semantic class index.
    """
    return np.argmax(image, axis = -1)

def colour_code_segmentation(image, label_values):
    """
    # This function is used to visulized the reverse_one_hot(array(output_tensor)).
    Given a 1-channel array of class indexs after reverse_one_hot, colouring the array 
    with corresponding label values.
    
    @image=[H,W]:
        single channel image array at t, after reverse_one_hot, where each value represents the class index.
    
    @label_values:
        A list of numerica semantic values of RGB for each sementic class.
        
    @return=[H,W,C=3]:
        Coloured image for segmentation visualization.
    """  
    colour_codes = np.array(label_values)
    
    return colour_codes[image.astype(int)]

def seq_name(num):
    """
    @num: 
        ordinal number of file
    
    @return:
        string(ordinal number)
    """

    length=5
    num_str=str(num)
    if length >= len(num_str):
        result = '0' * (length-len(num_str)) + num_str
    return result

################
################

#####
# Step 1: Customize arguments
#####

args = easydict.EasyDict({
    'indir_test_files':"./data/val_image_W3840/*.png",
    'indir_dataset': "./data",
    'indir_pretrained_weights': "./result/ckpt/latest_model.ckpt",
    'outdir_pretrained_weights': "./result/trained_weights.txt",
    'outdir_predition': "./result/pred/",
    'H': 256,
    'W':3840,
    'C':3,
    'stride': 2,
    'T':2160
})
H=args.H
W=args.W
C=args.C
s=args.stride
T=args.T

if not os.path.isdir(args.outdir_predition):
    os.makedirs(args.outdir_predition)
    
## read in information about semantic classes
class_names_list, label_values = get_label_info(os.path.join(args.indir_dataset, "class_dict.csv"))
num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.indir_dataset)
print("Num Classes -->", num_classes)


###############
## Step 2: extracting pre-trained weights from checkpoints of training backbone model
###############

reader = tf.train.NewCheckpointReader(args.indir_pretrained_weights)
shapes_dict = reader.get_variable_to_shape_map()

names = []
dimensions = []
for key, value in shapes_dict.items():
    names.append(key)
    dimensions.append(value)
    
# write into a txt file
file = open(args.outdir_pretrained_weights, "w") 
for i in range(len(names)):
    name = names[i]
    dimension = str(dimensions[i])
    file.write(name +  '  ')
    file.write(dimension)
    file.write('\n')
file.close()



###############
## Step 3: loading pre-trained weights
###############

## Initial Stage
w0 = reader.get_tensor('Conv/weights')        # [2, 2, C_in=3, C_out=32]
b0 = reader.get_tensor('Conv/biases')         # [C=32]

## Down-sampling
w1 = reader.get_tensor('Conv_1/weights')      # [2, 2, C_in=32, C_out=48]
b1 = reader.get_tensor('Conv_1/biases')       # [C=48]

w2 = reader.get_tensor('Conv_2/weights')      # [2, 2, C_in=48, C_out=72]
b2 = reader.get_tensor('Conv_2/biases')       # [C=96]

w3 = reader.get_tensor('Conv_3/weights')      # [2, 2, C_in=72, C_out=96]
b3 = reader.get_tensor('Conv_3/biases')       # [C=192]

w4 = reader.get_tensor('Conv_4/weights')      # [2, 2, C_in=96, C_out=192]
b4 = reader.get_tensor('Conv_4/biases')       # [C=256]

w5 = reader.get_tensor('Conv_5/weights')      # [2, 2, C_in=192, C_out=256]
b5 = reader.get_tensor('Conv_5/biases')       # [C=320]

w6 = reader.get_tensor('Conv_6/weights')      # [2, 2, C_in=256, C_out=320]
b6 = reader.get_tensor('Conv_6/biases')       # [C=384]

w7 = reader.get_tensor('Conv_7/weights')      # [2, 2, C_in=384+320, C_out=384]
b7 = reader.get_tensor('Conv_7/biases')       # [C=384]

## Up-sampling
w8 = reader.get_tensor('Conv_8/weights')      # [2, 2, C_in=384+256, C_out=320]
b8 = reader.get_tensor('Conv_8/biases')       # [C=320]

w9 = reader.get_tensor('Conv_9/weights')      # [2, 2, C_in=320+192, C_out=256]
b9 = reader.get_tensor('Conv_9/biases')       # [C=256]

w10 = reader.get_tensor('Conv_10/weights')    # [1, 2, C_in=256+96, C_out=192]
b10 = reader.get_tensor('Conv_10/biases')     # [C=192]

w11 = reader.get_tensor('Conv_11/weights')    # [1, 2, C_in=192+48, C_out=96]
b11 = reader.get_tensor('Conv_11/biases')     # [C=96]

w12 = reader.get_tensor('Conv_12/weights')    # [1, 2, C_in=96+32, C_out=48]
b12 = reader.get_tensor('Conv_12/biases')     # [C=48]

w13 = reader.get_tensor('Conv_13/weights')    # [1, 2, C_in=48, C_out=32]
b13 = reader.get_tensor('Conv_13/biases')     # [C=32]

w14 = reader.get_tensor('Conv_14/weights')   
b14 = reader.get_tensor('Conv_14/biases') 

w15 = reader.get_tensor('Conv_15/weights')   
b15 = reader.get_tensor('Conv_15/biases') 

# Softmax
w16 = reader.get_tensor('logits/weights')   
b16 = reader.get_tensor('logits/biases')    



###############
## Step 4: testing with SMN
###############


###############################################################################
### 4.2 Initializing the memory structure : 
# First shift nodes, then insert new node.
# Memory structure is updated after each pooling layer.
###############################################################################

re = np.zeros((T, W, C), dtype=np.uint8)
out_vis_image = np.zeros((2, W, C), dtype=np.uint8)

M0_p = np.zeros([1, (2**0)+1, W, C], np.float32)                # [1, H=2, W=2048, C=3]
M0_c = np.zeros([1, (2**0)+1, W, 32], np.float32)               # [1, H=2, W=2048, C=32]
M0_c = tf.Variable(M0_c)

M1_p = np.zeros([1, (2**1)+1, W//(2**1), 32], np.float32)       # [1, H=3, W=1024, C=32]
M1_p = tf.Variable(M1_p)
M1_c = np.zeros([1, (2**1)+1, W//(2**1), 48], np.float32)       # [1, H=3, W=1024, C=48]
M1_c = tf.Variable(M1_c)

M2_p = np.zeros([1, (2**2)+1, W//(2**2), 48], np.float32)       # [1, H=5, W=512, C=48]
M2_p = tf.Variable(M2_p)
M2_c = np.zeros([1, (2**2)+1, W//(2**2), 96], np.float32)       # [1, H=5, W=512, C=72]
M2_c = tf.Variable(M2_c)

M3_p = np.zeros([1, (2**3)+1, W//(2**3), 96], np.float32)       # [1, H=9, W=256, C=72]
M3_p = tf.Variable(M3_p)
M3_c = np.zeros([1, (2**3)+1, W//(2**3), 128], np.float32)       # [1, H=9, W=256, C=96]
M3_c = tf.Variable(M3_c)

M4_p = np.zeros([1, (2**4)+1, W//(2**4), 128], np.float32)       # [1, H=17, W=128, C=96]
M4_p = tf.Variable(M4_p)
M4_c = np.zeros([1, (2**4)+1, W//(2**4), 192], np.float32)      # [1, H=17, W=128, C=128]
M4_c = tf.Variable(M4_c)

M5_p = np.zeros([1, (2**5)+1, W//(2**5), 192], np.float32)      # [1, H=33, W=64, C=128]
M5_p = tf.Variable(M5_p)
M5_c = np.zeros([1, (2**5)+1, W//(2**5), 256], np.float32)      # [1, H=33, W=64, C=192]
M5_c = tf.Variable(M5_c)

M6_p = np.zeros([1, (2**6)+1, W//(2**6), 256], np.float32)      # [1, H=65, W=32, C=192]
M6_p = tf.Variable(M6_p)
M6_c = np.zeros([1, (2**6)+1, W//(2**6), 320], np.float32)      # [1, H=65, W=32, C=256]
M6_c = tf.Variable(M6_c)

M7_p = np.zeros([1, (2**7)+1, W//(2**7), 320], np.float32)      # [1, H=129, W=16, C=256]
M7_p = tf.Variable(M7_p)
M7_c = np.zeros([1, (2**7)+1, W//(2**7), 384], np.float32)      # [1, H=129, W=16, C=320]
M7_c = tf.Variable(M7_c)

###################################################################



##################################
#### Testing
################################## 
files = glob(args.indir_test_files)
for file in files: 
    tf.reset_default_graph()    # clear all previous sessions
    
    rp = cv2.imread(file)              
    rp = cv2.cvtColor(rp, cv2.COLOR_BGR2RGB)
    rp = np.float32(rp)/255.0  
    
    ### SMN algorithm
    #########################################
    t0 = time.time()
    for i in range(0, T):
      
      M0_p[0][0][:][:] = M0_p[0][1][:][:]
      M0_p[0][-1][:][:] = rp[i][:][:]
      
      
      ###################
      #### Initialization
      ###################  
      T_new = i%2
      T_old = (T_new-2**0)%2     
      #conv
      net = tf.nn.relu(conv(M0_p, w0, b0), name='ReLU')                              
      #update-conv  
      M0_c[:, T_new, :, :].assign(net[:, 0, :, :])

      ##################
      #### Down-sampling
      ##################
       
      ### Encoding Block 1
      T_new = i%3
      T_old = (T_new-2**1)%3
      #pool
      net = slim.pool(M0_c, [s, s], stride=[s, s], pooling_type='MAX')
      #update-pool
      M1_p[:, T_new, :, :].assign(net[:, 0, :, :])
      #prepare-conv
      net1 = tf.concat([tf.slice(M1_p, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M1_p, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
      #conv
      net = tf.nn.relu(conv(net1, w1, b1), name='ReLU')
      #update-conv  
      M1_c[:, T_new, :, :].assign(net[:, 0, :, :])           
      #prepare-pool 
      net = tf.concat([tf.slice(M1_c, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M1_c, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
 
      ### Encoding Block 2
      T_new = i%5
      T_old = (T_new-2**2)%5
      #pool
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      #update-pool
      M2_p[:, T_new, :, :].assign(net[:, 0, :, :])
      #prepare-conv
      net2 = tf.concat([tf.slice(M2_p, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M2_p, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
      #conv
      net = tf.nn.relu(conv(net2, w2, b2), name='ReLU')
      #update-conv             
      M2_c[:, T_new, :, :].assign(net[:, 0, :, :])     
      #prepare-pool 
      net = tf.concat([tf.slice(M2_c, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M2_c, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
 
      ### Encoding Block 3
      T_new = i%9
      T_old = (T_new-2**3)%9
      #pool
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      #update-pool
      M3_p[:, T_new, :, :].assign(net[:, 0, :, :])
      #prepare-conv
      net3 = tf.concat([tf.slice(M3_p, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M3_p, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
      #conv
      net = tf.nn.relu(conv(net3, w3, b3), name='ReLU')
      #update-conv             
      M3_c[:, T_new, :, :].assign(net[:, 0, :, :])     
      #prepare-pool 
      net = tf.concat([tf.slice(M3_c, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M3_c, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
 
      ### Encoding Block 4
      T_new = i%17
      T_old = (T_new-2**4)%17
      #pool
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      #update-pool
      M4_p[:, T_new, :, :].assign(net[:, 0, :, :])
      #prepare-conv
      net4 = tf.concat([tf.slice(M4_p, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M4_p, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
      #conv
      net = tf.nn.relu(conv(net4, w4, b4), name='ReLU')
      #update-conv             
      M4_c[:, T_new, :, :].assign(net[:, 0, :, :])     
      #prepare-pool 
      net = tf.concat([tf.slice(M4_c, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M4_c, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
 
      ### Encoding Block 5
      T_new = i%33
      T_old = (T_new-2**5)%33
      #pool
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      #update-pool
      M5_p[:, T_new, :, :].assign(net[:, 0, :, :])
      #prepare-conv
      net5 = tf.concat([tf.slice(M5_p, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M5_p, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
      #conv
      net = tf.nn.relu(conv(net5, w5, b5), name='ReLU')
      #update-conv             
      M5_c[:, T_new, :, :].assign(net[:, 0, :, :])     
      #prepare-pool 
      net = tf.concat([tf.slice(M5_c, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M5_c, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
 
      ### Encoding Block 6
      T_new = i%65
      T_old = (T_new-2**6)%65
      #pool
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      #update-pool
      M6_p[:, T_new, :, :].assign(net[:, 0, :, :])
      #prepare-conv
      net6 = tf.concat([tf.slice(M6_p, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M6_p, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
      #conv
      net = tf.nn.relu(conv(net6, w6, b6), name='ReLU')
      #update-conv             
      M6_c[:, T_new, :, :].assign(net[:, 0, :, :])     
      #prepare-pool 
      net = tf.concat([tf.slice(M6_c, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M6_c, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
       
      ### Encoding Block 7
      T_new = i%129
      T_old = (T_new-2**7)%129
      #pool
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      #update-pool
      M7_p[:, T_new, :, :].assign(net[:, 0, :, :])
      #prepare-conv
      net7 = tf.concat([tf.slice(M7_p, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M7_p, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
      #conv
      net = tf.nn.relu(conv(net7, w7, b7), name='ReLU')
      #update-conv             
      M7_c[:, T_new, :, :].assign(net[:, 0, :, :])     
      #prepare-pool 
      net = tf.concat([tf.slice(M7_c, [0,T_old,0,0], [-1,1,-1,-1]), tf.slice(M7_c, [0,T_new,0,0], [-1,1,-1,-1])], axis=1)
     
      ### Encoding Block 8
      #pool
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')

 
      ##############
      ## Up-sampling
      ##############
        
      ### Decoding Block 1
      net = layers.UpSampling2D(size=(s,s))(net)
      net = tf.concat([net, net7], axis=-1)      # [1, H=2, W=16, C=384+256=640]
      net = tf.nn.relu(conv(net, w8, b8), name='ReLU') 
      
      ### Decoding Block 2
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))
      net = tf.concat([net, net6], axis=-1)       # [1, H=2, W=32, C=384+192=576]
      net = tf.nn.relu(conv(net, w9, b9), name='ReLU')
      
      ### Decoding Block 3
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))  
      net = tf.concat([net, net5], axis=-1)       # [1, H=2, W=64, C=320+128=448]
      net = tf.nn.relu(conv(net, w10, b10), name='ReLU')
      
      ### Decoding Block 4
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))       # [1, H=2, W=128, C=256]  
      net = tf.concat([net, net4], axis=-1)       # [1, H=1, W=128, C=256+96=352]
      net = tf.nn.relu(conv(net, w11, b11), name='ReLU')
      
      ### Decoding Block 5
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))       # [1, H=2, W=256, C=192]        
      net = tf.concat([net, net3], axis=-1)       # [1, H=2, W=256, C=192+72=264]
      net = tf.nn.relu(conv(net, w12, b12), name='ReLU')

      ### Decoding Block 6
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))      # [1, H=2, W=512, C=128]  
      net = tf.concat([net, net2], axis=-1)       # [1, H=2, W=512, C=128+48=176]
      net = tf.nn.relu(conv(net, w13, b13), name='ReLU')

      ### Decoding Block 7
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))      # [1, H=2, W=1024, C=96]  
      net = tf.concat([net, net1], axis=-1)        # [1, H=2, W=1024, C=96+32=128]
      net = tf.nn.relu(conv(net, w14, b14), name='ReLU')

      ### Decoding Block 8
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))     # [1, H=2, W=2048, C=72]  
      net = tf.concat([net, tf.Variable(M0_p)], axis=-1)       # [1, H=2, W=2048, C=72+3=75]
      net = tf.nn.relu(conv(net, w15, b15), name='ReLU')
      
      ### softmax
      net= tf.nn.softmax(conv(net, w16, b16), axis=-1, name=None)
      #######
          
      net = net.numpy()                         
      net = net[0, :,:,:]                         # [H=2, W=2048, C=num_class]
      out_image = reverse_one_hot(net)            # [H=2, W=2048] 
      out_image = cv2.cvtColor(np.uint8(colour_code_segmentation(out_image, label_values)), cv2.COLOR_RGB2BGR)   # [H=2, W=2048, C=3]
      
      re[i, :, :] = out_image[1, :, :] 
      
      #print('Frame_'+str(i))
      
    #########################################
    
    t1 = time.time() - t0
    name = os.path.basename(file)
    name = os.path.splitext(name)[0] + '.png'
    cv2.imwrite(args.outdir_predition+name, re)
    print('TestTime for ' + name + ' = ' + str(t1) + ' s')