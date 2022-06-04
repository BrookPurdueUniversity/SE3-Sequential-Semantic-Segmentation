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
def shift(arr):
    '''
    for i in range(0, arr.shape[1]-1):
        arr[0][i][:][:] = arr[0][i+1][:][:]  
    return arr
    '''

    '''
    ## Example
    T=8
    H=T+1  # H=9
    arr = [0,1,2,3,4,5,6,7,8]  # H=T+1

    # body
    for i in reversed(range(1, H-1)):
        arr[i] = arr[i-1]
        print('i='+str(i)+':'+str(arr))
    # 1st node    
    arr[0] = arr[H-1]
    '''
    
    '''
    # body
    H=arr.shape[1]  # H=T+1
    for i in reversed(range(1, H-1)):  # T, T-1, ..., 1
       arr[0][i][:][:] = arr[0][i-1][:][:]
    # 1st node
    arr[0][0][:][:] = arr[0][H-1][:][:]
    # last node: empty
    return arr
    '''

    H=arr.shape[1]                 # H=T+1
    for i in (range(1, H-1)):      # 1. 2… T-1
       arr[0][i][:][:] = arr[0][i + 1][:][:]
    # last node
    arr[0][H - 1][:][:]= arr[0][0][:][:]
    # first node: empty
    return arr

def conv(tensor, weights, biases): 
    conv = tf.nn.conv2d(tensor, weights, strides=[1,1,1,1], padding='SAME')
    output = conv + biases
    return output

def get_label_info(csv_path):
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

def reverse_one_hot(image):
    return np.argmax(image, axis = -1)

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    return colour_codes[image.astype(int)]

###############################################################################
###############################################################################

#############################
# Step 1: Customize arguments
#############################

args = easydict.EasyDict({
    'indir_test_files':"./data/val_image_W4096/*.png",
    'indir_dataset': "./data",
    'indir_pretrained_weights': "./result/ckpt/latest_model.ckpt",
    'outdir_pretrained_weights': "./result/trained_weights.txt",
    'outdir_predition': "./result/pred_0420/",
    'H': 256,
    'W': 4096, #3840,
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


##########################################################
## Step 2: extracting pre-trained weights from checkpoints
##########################################################

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


######################################
## Step 3: loading pre-trained weights
######################################

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
w16 = reader.get_tensor('logits/weights')   # [1, 1, C_in=32, C_out=num_classes]
b16 = reader.get_tensor('logits/biases')    # [C=num_classes]


###########################
## Step 4: testing with SMN
###########################

re = np.zeros((T, W, C), dtype=np.uint8)
out_vis_image = np.zeros((2, W, C), dtype=np.uint8)

M0_p = np.zeros([1, (2**0)+1, W, C], np.float32)                # [1, H=2, W=1024, C=3]
M0_c = np.zeros([1, (2**0)+1, W, 32], np.float32)               # [1, H=2, W=1024, C=32]

M1_p = np.zeros([1, (2**1)+1, W//(2**1), 32], np.float32)       # [1, H=3, W=512, C=32]
M1_c = np.zeros([1, (2**1)+1, W//(2**1), 48], np.float32)       # [1, H=3, W=512, C=48]

M2_p = np.zeros([1, (2**2)+1, W//(2**2), 48], np.float32)       # [1, H=5, W=256, C=48]
M2_c = np.zeros([1, (2**2)+1, W//(2**2), 96], np.float32)       # [1, H=5, W=256, C=96]

M3_p = np.zeros([1, (2**3)+1, W//(2**3), 96], np.float32)       # [1, H=9, W=128, C=96]
M3_c = np.zeros([1, (2**3)+1, W//(2**3), 128], np.float32)       # [1, H=9, W=128, C=192]

M4_p = np.zeros([1, (2**4)+1, W//(2**4), 128], np.float32)       # [1, H=17, W=64, C=192]
M4_c = np.zeros([1, (2**4)+1, W//(2**4), 192], np.float32)      # [1, H=17, W=64, C=256]

M5_p = np.zeros([1, (2**5)+1, W//(2**5), 192], np.float32)      # [1, H=33, W=32, C=256]
M5_c = np.zeros([1, (2**5)+1, W//(2**5), 256], np.float32)      # [1, H=33, W=32, C=320]

M6_p = np.zeros([1, (2**6)+1, W//(2**6), 256], np.float32)      # [1, H=65, W=16, C=320]
M6_c = np.zeros([1, (2**6)+1, W//(2**6), 320], np.float32)      # [1, H=65, W=16, C=384]

M7_p = np.zeros([1, (2**7)+1, W//(2**7), 320], np.float32)      # [1, H=65, W=16, C=320]
M7_c = np.zeros([1, (2**7)+1, W//(2**7), 384], np.float32)      # [1, H=65, W=16, C=384]


## Loading testing images
files = glob(args.indir_test_files)
for file in files:    
    tf.reset_default_graph()     # clear all previous sessions
    
    rp = cv2.imread(file)       # [H=2160, W=3840, C=3]
    rp = cv2.cvtColor(rp, cv2.COLOR_BGR2RGB)
    rp = np.float32(rp)/255.0   
    
    ### SMN algorithm
    #########################################
    t0 = time.time()
    for i in reversed(range(0, T)):
      
      # SHIFT the memory at each layer 1-H-unit
      M0_p[0][1][:][:] = M0_p[0][0][:][:] 
      M1_p = shift(M1_p)      
      M2_p = shift(M2_p)      
      M3_p = shift(M3_p)      
      M4_p = shift(M4_p)  
      M5_p = shift(M5_p) 
      M6_p = shift(M6_p) 
      M7_p = shift(M7_p)
      
      #M0_c[0][0][:][:] = M0_c[0][1][:][:]
      M0_c[0][1][:][:] = M0_c[0][0][:][:]    
      
      M1_c = shift(M1_c)      
      M2_c = shift(M2_c)      
      M3_c = shift(M3_c)      
      M4_c = shift(M4_c) 
      M5_c = shift(M5_c) 
      M6_c = shift(M6_c)  
      M7_c = shift(M7_c)          
      
      #M0_p[0][-1][:][:] = rp[i][:][:]  
      M0_p[0][0][:][:] = rp[i][:][:]
      
      ###################
      #### Initialization
      ###################
      #conv
      net = conv(M0_p, w0, b0)                          # [1, H=2, W=1024, C=32]
      net = tf.nn.relu(net, name='ReLU')
      #update-conv           
      M0_c[0][0][:][:] = (net.numpy())[0][0][:][:]     
      
      ##################
      #### Down-sampling
      ##################
      ### Encoding Block 1
      #pool
      net = slim.pool(M0_c, [s, s], stride=[s, s], pooling_type='MAX')     # [1, H=1, W=512, C=32]                                
      #update-pool
      M1_p[0][0][:][:] = (net.numpy())[0][:][:][:]     
      #conv
      net = conv(M1_p[:, :2, :,:], w1, b1)             # [1, H=2, W=512, C=48]
      net = tf.nn.relu(net, name='ReLU') 
      #update-conv             
      M1_c[0][0][:][:] = (net.numpy())[0][0][:][:]     

      ### Encoding Block 2
      net = slim.pool(M1_c[:, :2, :, :], [s, s], stride=[s, s], pooling_type='MAX')
      M2_p[0][0][:][:] = (net.numpy())[0][:][:][:] 
      net = conv(M2_p[:, :2, :,:], w2, b2)             # [1, H=2, W=256, C=96]
      net = tf.nn.relu(net, name='ReLU')          
      M2_c[0][0][:][:] = (net.numpy())[0][0][:][:]     
      
      ### Encoding Block 3
      net = slim.pool(M2_c[:, :2, :, :], [s, s], stride=[s, s], pooling_type='MAX')     
      M3_p[0][0][:][:] = (net.numpy())[0][:][:][:] 
      net = conv(M3_p[:, :2, :,:], w3, b3)             # [1, H=2, W=128, C=128] 
      net = tf.nn.relu(net, name='ReLU')           
      M3_c[0][0][:][:] = (net.numpy())[0][0][:][:]     
      
      ### Encoding Block 4
      net = slim.pool(M3_c[:, :2, :, :], [s, s], stride=[s, s], pooling_type='MAX')
      M4_p[0][0][:][:] = (net.numpy())[0][:][:][:] 
      net = conv(M4_p[:, :2, :,:], w4, b4)             # [1, H=2, W=64, C=192]
      net = tf.nn.relu(net, name='ReLU')            
      M4_c[0][0][:][:] = (net.numpy())[0][0][:][:]  

      ### Encoding Block 5
      net = slim.pool(M4_c[:, :2, :, :], [s, s], stride=[s, s], pooling_type='MAX')
      M5_p[0][0][:][:] = (net.numpy())[0][:][:][:] 
      net = conv(M5_p[:, :2, :,:], w5, b5)             # [1, H=2, W=32, C=256]
      net = tf.nn.relu(net, name='ReLU')            
      M5_c[0][0][:][:] = (net.numpy())[0][0][:][:]  

      ### Encoding Block 6
      net = slim.pool(M5_c[:, :2, :, :], [s, s], stride=[s, s], pooling_type='MAX')
      M6_p[0][0][:][:] = (net.numpy())[0][:][:][:] 
      net = conv(M6_p[:, :2, :,:], w6, b6)             # [1, H=2, W=16, C=320]
      net = tf.nn.relu(net, name='ReLU')            
      M6_c[0][0][:][:] = (net.numpy())[0][0][:][:]  

      ### Encoding Block 7
      net = slim.pool(M6_c[:, :2, :, :], [s, s], stride=[s, s], pooling_type='MAX')
      M7_p[0][0][:][:] = (net.numpy())[0][:][:][:] 
      net = conv(M7_p[:, :2, :,:], w7, b7)             # [1, H=2, W=8, C=384]
      net = tf.nn.relu(net, name='ReLU')            
      M7_c[0][0][:][:] = (net.numpy())[0][0][:][:]  
      
      ### Encoding Block 8
      net = slim.pool(M7_c[:, :2, :, :], [s, s], stride=[s, s], pooling_type='MAX')   # [1, H=1, W=4, C=384] 
                                                                     
           
      ##############
      ## Up-sampling
      ##############
        
      ### Decoding Block 1
      net = layers.UpSampling2D(size=(s,s))(net)       # [1, H=2, W=8, C=384]
      net = tf.concat([net, M7_p[:, :2, :,:]], axis=-1)           # [1, H=2, W=8, C=384+320]
      net = conv(net, w8, b8)                          # [1, H=2, W=8, C=384]
      net = tf.nn.relu(net, name='ReLU')    
      net = tf.slice(net, [0,0,0,0], [-1,1,-1,-1])    # [1, H=1, W=8, C=384] 
      
      ### Decoding Block 2
      net = layers.UpSampling2D(size=(s,s))(net)       # [1, H=2, W=16, C=384]       
      net = tf.concat([net, M6_p[:, :2, :,:]], axis=-1)           # [1, H=2, W=16, C=384+256]
      net = conv(net, w9, b9)                          # [1, H=2, W=16, C=320]
      net = tf.nn.relu(net, name='ReLU') 
      net = tf.slice(net, [0,0,0,0], [-1,1,-1,-1])
      
      ### Decoding Block 3
      net = layers.UpSampling2D(size=(s,s))(net)       # [1, H=2, W=32, C=320]        
      net = tf.concat([net, M5_p[:, :2, :,:]], axis=-1)           # [1, H=2, W=32, C=320+192]
      net = conv(net, w10, b10)                        # [1, H=2, W=32, C=256]
      net = tf.nn.relu(net, name='ReLU')  
      net = tf.slice(net, [0,0,0,0], [-1,1,-1,-1])
      
      ### Decoding Block 4
      net = layers.UpSampling2D(size=(s,s))(net)       # [1, H=2, W=64, C=256]  
      net = tf.concat([net, M4_p[:, :2, :,:]], axis=-1)           # [1, H=1, W=64, C=256+128]
      net = conv(net, w11, b11)                        # [1, H=1, W=64, C=192]
      net = tf.nn.relu(net, name='ReLU')  
      net = tf.slice(net, [0,0,0,0], [-1,1,-1,-1])

      ### Decoding Block 5
      net = layers.UpSampling2D(size=(s,s))(net)       # [1, H=2, W=128, C=192]  
      net = tf.concat([net, M3_p[:, :2, :,:]], axis=-1)           # [1, H=1, W=128, C=192+96]
      net = conv(net, w12, b12)                        # [1, H=1, W=128, C=128]
      net = tf.nn.relu(net, name='ReLU')  
      net = tf.slice(net, [0,0,0,0], [-1,1,-1,-1])
      
      ### Decoding Block 6
      net = layers.UpSampling2D(size=(s,s))(net)       # [1, H=2, W=256, C=128]        
      net = tf.concat([net, M2_p[:, :2, :,:]], axis=-1)           # [1, H=2, W=256, C=128+48]
      net = conv(net, w13, b13)                        # [1, H=2, W=256, C=96]
      net = tf.nn.relu(net, name='ReLU')  
      net = tf.slice(net, [0,0,0,0], [-1,1,-1,-1])

      ### Decoding Block 7
      net = layers.UpSampling2D(size=(s,s))(net)       # [1, H=2, W=512, C=96]  
      net = tf.concat([net, M1_p[:, :2, :,:]], axis=-1)           # [1, H=2, W=512, C=96+32]      
      net = conv(net, w14, b14)                        # [1, H=2, W=512, C=48]
      net = tf.nn.relu(net, name='ReLU') 
      net = tf.slice(net, [0,0,0,0], [-1,1,-1,-1])

      ### Decoding Block 8
      net = layers.UpSampling2D(size=(s,s))(net)       # [1, H=2, W=1024, C=48]  
      net = tf.concat([net, M0_p], axis=-1)           # [1, H=2, W=1024, C=48+3]      
      net = conv(net, w15, b15)                        # [1, H=2, W=1024, C=32]
      net = tf.nn.relu(net, name='ReLU') 
      
      ### softmax
      net = conv(net, w16, b16)                        # [1, H=2, W=1024, C=num_class]
      net = tf.nn.softmax(net, axis=-1, name=None)                                
      #######
      
      net = tf.squeeze(net, 0)              
      net = net.numpy()                                # [1, H=2, W=3804, C=num_class]                           
      output_image = reverse_one_hot(net)              # [H=2, W=3804] 
      out_vis_image = cv2.cvtColor(np.uint8(colour_code_segmentation(output_image, label_values)), cv2.COLOR_RGB2BGR)   # [H=2, W=3840, C=3]
      
      re[i, :, :] = out_vis_image[0, :, :] 
      
    #########################################

    t1 = time.time() - t0
    name = os.path.basename(file)
    name = os.path.splitext(name)[0] + '.png'
    cv2.imwrite(args.outdir_predition+name, re)
    print('TestTime for ' + name + ' = ' + str(t1) + ' s')