import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
# from dataset.augmentation import AllAugmentationTransform
# from augmentation import AllAugmentationTransform

import glob
import cv2
import torch
import random

class BiWiDataset(Dataset):
    IMAGE_SIZE = 256
    def __init__(self, data_dir, transform=False):
        self.data_dir = data_dir
        self.pngs = glob.glob(f'{data_dir}/faces_0/*/*_cropface.png')
        self.transform = transform

    def __len__(self):
        return len(self.pngs)

    def __load_label(self, txt, out_shape=(68,2)):
        if os.path.isfile(txt):
            with open(txt, 'r') as f:
                line = f.readline()
                line =  np.array([float(l) for l in line.split(",")])
                line = line.reshape(out_shape)
                return line
        else:
            return None

    def __norm_lmks(self, lmks):
        lmks = (lmks*2.0)/self.IMAGE_SIZE - 1.0 # -1 -> 1
        return lmks

    def __getitem__(self, index):
        src_png = self.pngs[index]
        img_src = cv2.imread(src_png)

        lmks_src = self.__load_label(f'{src_png[:-13]+"_lmksface_ibus.txt"}', out_shape=(68,2))
        pose_src = self.__load_label(f'{src_png[:-13]+"_pose_ibus.txt"}', out_shape=(-1, 3))

        # Find a random file in user folder
        user_dir = os.path.dirname(src_png)
        dst_png = random.choice(glob.glob(f'{user_dir}/*_cropface.png'))

        img_drv = cv2.imread(dst_png)
        lmks_drv = self.__load_label(f'{dst_png[:-13]+"_lmksface_ibus.txt"}', out_shape=(68,2))
        pose_drv = self.__load_label(f'{dst_png[:-13]+"_pose_ibus.txt"}', out_shape=(-1, 3))

        # import pdb; pdb.set_trace()

        if lmks_src is  None or lmks_drv is  None:
            return None

        if self.transform:
            scale = self.IMAGE_SIZE / img_src.shape[1]
            lmks_src = lmks_src * scale
            lmks_drv = lmks_drv * scale

            img_src = cv2.resize(img_src, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            img_drv = cv2.resize(img_drv, (self.IMAGE_SIZE, self.IMAGE_SIZE))

            img_src = img_src / 255.0
            img_src = torch.FloatTensor(img_src).permute(2,0,1)

            img_drv = img_drv / 255.0
            img_drv = torch.FloatTensor(img_drv).permute(2,0,1)

            lmks_src = self.__norm_lmks(lmks_src)
            lmks_src = torch.FloatTensor(lmks_src)
            lmks_drv = self.__norm_lmks(lmks_drv)
            lmks_drv = torch.FloatTensor(lmks_drv)

            pose_src = torch.FloatTensor(pose_src)
            pose_drv = torch.FloatTensor(pose_drv)


        # return {"source":img_src, "driving":img_drv, "lmks_source":{"value": lmks_src}, "lmks_driving":{"value":lmks_drv}, "pose_src":pose_src, "pose_drv":pose_drv}
        return {"source":img_src, "driving":img_drv, "lmks_source":{"value": lmks_src}, "lmks_driving":{"value":lmks_drv}, "name": src_png}



def denorm_lmks(lmks):
    # From -1, 1 => normal range
    lmks = ((lmks+1)*256)/2.0
    return lmks   

def draw_landmarks(img, lmks, color=(255,0,0)):
    img = np.ascontiguousarray(img)
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, color, -1, lineType=cv2.LINE_AA)

    return img


if __name__ == "__main__":
    dset = BiWiDataset(data_dir="/home/ubuntu/vuthede/public-dataset/biwi", transform=True)
    print(f'Len dset :{len(dset)}')
    
    output_debug = "debug_biwi"
    if not os.path.isdir(output_debug):
        os.makedirs(output_debug)

    for i in range(100):
        data = dset[i]
        img_src = data["source"].permute(1,2,0).numpy()*255.0
        img_drv = data["driving"].permute(1,2,0).numpy()*255.0
        lmks_src = denorm_lmks(data["lmks_source"]["value"].numpy())
        lmks_drv = denorm_lmks(data["lmks_driving"]["value"].numpy())

        img_src = draw_landmarks(img_src, lmks_src)
        img_drv = draw_landmarks(img_drv, lmks_drv)

        concat_img = np.hstack([img_src, img_drv])
        cv2.imwrite(f'{output_debug}/{i}.png', concat_img)

    # import pdb; pdb.set_trace()
    print("End")