from glob import glob
import cv2


files_gt = glob('./data/val_label/*.png')
N = len(files_gt)

files_pred = glob('./result/smn_fast/*.png')

dir_output = './Evaluation/smn_fast_mIoU.txt'

RGBs=[[0,0,0], [128,0,0], [128,64,128], [0,128,0], [128,128,0], [64,0,128], [192,0,192], [64,64,0]]
U = [0,0,0,0,0,0,0,0]
I = [0,0,0,0,0,0,0,0]


init = 256*2

for k in range(N):
    
    gt = cv2.imread(files_gt[k])
    gt = gt[init:,:,:]
    pred = cv2.imread(files_pred[k])
    pred = pred[init:,:,:]
    
    # scan per pix
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):  
            
            # gt_id
            r = gt[i][j][2] 
            g = gt[i][j][1]
            b = gt[i][j][0]
            id_gt = RGBs.index([r,g,b])
            
            U[id_gt] += 1
            
            # pred_id
            r = pred[i][j][2] 
            g = pred[i][j][1]
            b = pred[i][j][0]
            id_pred = RGBs.index([r,g,b])  
                       
            if id_pred==id_gt:
                I[id_gt] += 1
            else:
                U[id_pred] += 1
                
# IoU for single Image            
IoU = []
for i in range(len(U)):
    IoU.append(I[i]/U[i])
    print('The IoU of' + str(i) + 'th class=', IoU[i])

        
# write into txt
file = open(dir_output, "w") 
for i in range(len(IoU)):
    file.write(str(IoU[i])) 
    file.write('\n')
file.close()      