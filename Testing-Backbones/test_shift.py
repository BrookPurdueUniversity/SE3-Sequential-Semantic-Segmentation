import helpers
import model
###############################################################################
import os,cv2,time
import tensorflow as tf
import numpy as np
import easydict
tf.reset_default_graph()
from glob import glob


#####
# Step 1: Customize arguments
# Overall functions should at least contain: 1-loading pretrained model, 2-testing.
#####

args = easydict.EasyDict({
    'test_image':"./data/val_image/*.png",
    'checkpoint_path': './result/ckpt/latest_model.ckpt',
    'dataset': "./data",
    'predict_result_dir': "./result/pred/",
    'H': 256,
    'T':2160,
})
H = args.H
T = args.T

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
num_classes = len(label_values)


#####
# Step 2: Initialize the network
#####
# building up tf.Session, which is necessary in tensorflow.__version__=1
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])  # initialize input tensor, which must be at least 4D
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])  # initialize output tensor, which must be at least 4D
network = model.build(inputs=net_input, num_classes=num_classes)  # recall customized model
saver=tf.train.Saver(max_to_keep=1000)   # declare the saver container

sess.run(tf.global_variables_initializer())   # make sure all variables started from initializetion
saver.restore(sess, args.checkpoint_path)   # loading pretrained model

# Create directories for prediction results if visulization is needed
predict_dir = args.predict_result_dir
if not os.path.isdir(predict_dir):
    os.makedirs(predict_dir)

#####
# Step 3: Prediction
#####
print("\n********************* Begin prediction ****************************") 
files = glob(args.test_image)
for file in files:
    rp = cv2.imread(file)
    re = np.zeros_like(rp)
    rp = cv2.cvtColor(rp, cv2.COLOR_BGR2RGB)
    rp = np.expand_dims(np.float32(rp),axis=0)/255.0
    
    # Shift-Mode
    t0 = time.time()
    for i in range(0, T-H+1):
        input_image = rp[:, i:i+H, :, :]

        output_image = sess.run(network,feed_dict={net_input:input_image})        
        
        output_image = helpers.reverse_one_hot(np.array(output_image[0,:,:,:]))       
        output_vis_image = helpers.colour_code_segmentation(output_image, label_values)  
        output_vis_image = cv2.cvtColor(np.uint8(output_vis_image), cv2.COLOR_RGB2BGR)  # [H=32, W=256, C=3]
        

        # shift-mode with the last line
        if i==0:
            re[0:H, :, :] = output_vis_image
        else:
            re[H-1+i, :, :] = output_vis_image[-1, :, :] 
        
        '''
        # shift-mode with the middle line
        if i==0:
            re[0:H, :, :] = output_vis_image
        elif i==T-H:
            re[T-H:T, :, :] = output_vis_image
        else:
            re[H//2+i, :, :] = output_vis_image[H//2, :, :]  
        '''
        
        '''
        # shift-mode with the first line
        if i==T-H:
            re[T-H:T, :, :] = output_vis_image
        else:
            re[0+i, :, :] = output_vis_image[0, :, :] 
        '''
        
    t1 = time.time() - t0
    name = os.path.basename(file)
    name = os.path.splitext(name)[0] + '.png'
    cv2.imwrite(predict_dir + name, re)        
    print('TestTime for ' + name + ' = ' + str(t1) + ' s')

