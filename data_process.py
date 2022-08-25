import os, glob, numpy as np
import cv2 as cv

data_folder_path = '/home/hmcl/carla-birdeye-view/bev_datagen/carla-birdeye-view/autodata'
data_save_path = '/home/hmcl/AE-convLSTM/AE-convLSTM_AIFIT/dataset/'

files = glob.glob(data_folder_path)
cnt = 0
print("Start Data Processing...")
for folder in glob.glob(f'{data_folder_path}/*'):
    n_sim = folder[len(data_folder_path + '/'):len(folder)]
    X = []
    # frames_per_sim[cnt].append(int(n_sim))
    # frames_per_sim[cnt].append(len(glob.glob(f'{folder}/*.png')))
    # for num in range(len(glob.glob(f'{folder}/*.png'))):
    #     filename = folder + '/' + str((num+1)*2-1) + '.png'
    #     img = cv.imread(filename, cv.IMREAD_COLOR)
    #     data = np.asarray(img)
    #     X.append(data)
    # 14 장 기준으로 npz 저장
    if len(glob.glob(f'{folder}/*.png')) == 0:
        continue
    for num in range(14):
        filename = folder + '/' + str((num+1)*2-1) + '.png'
        img = cv.imread(filename, cv.IMREAD_COLOR)
        data = np.asarray(img)
        X.append(data)
        cnt += 1    
    # if n_sim == 63:
    #     print("here")
    np.savez(data_save_path + str(n_sim) + '.npz', X)



print("END Data Processing...")

            




