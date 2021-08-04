import os
from shutil import copyfile
import numpy as np
import os.path

download_path = 'known_image'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = 'known_images/'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

#train_val
train_path = download_path
photo_path = download_path + '/photo'
encoding_path = download_path + '/encoding'
capture_path = download_path + '/capture'
if not os.path.isdir(train_save_path):
    os.mkdir(photo_path)
    os.mkdir(encoding_path)
    os.mkdir(capture_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    if len(files) < 1:
       continue
    for i in range(len(files)):
        name = files[i]
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = root + "/" + name
        print(src_path)
        if (i > len(files)/2):
           dst_path = val_save_path
        else:
           dst_path = train_save_path
        cekfolder = dst_path.split("/")
        if cekfolder[1] == "train":
            if not os.path.isdir(dst_path+'/' + ID[0]+"_"+ID[1]+"/"):
               os.mkdir(dst_path+ '/' + ID[0]+"_"+ID[1]+"/")
            copyfile(src_path, dst_path + '/' + ID[0]+"_"+ID[1]+"/"+ name)
        else:
            if not os.path.isdir(dst_path):
               os.mkdir(dst_path)
            copyfile(src_path,dst_path + '/' + name)
