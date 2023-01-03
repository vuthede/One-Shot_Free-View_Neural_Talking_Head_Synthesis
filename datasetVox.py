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

import albumentations as A

transformerr = A.Compose(
    [
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.7),
        # A.ShiftScaleRotate (shift_limit=0.1, scale_limit=(-0.1,0.1), rotate_limit=5, interpolation=1, border_mode=1, always_apply=False, p=1)
        A.HorizontalFlip(p=0.5),
    ]
)


class VoxDataset(Dataset):
    IMAGE_SIZE = 256
    def __init__(self, data_dir, transform=False):
        self.data_dir = data_dir
        self.videos = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.videos)

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
        vid = f'{self.data_dir}/{self.videos[index]}'
        frame_ids = os.listdir(f'{vid}/img')
        frame_idx = np.sort(np.random.choice(frame_ids, replace=False, size=2))
        frame_idx = [f[:-4] for f in frame_idx] # 10.png -> 10

        src_name = frame_idx[0]
        drv_name = frame_idx[1]
        img_dir = vid + "/img"
        headpose_dir = vid + "/headpose_ibus"
        landmark_dir = vid + "/landmark_ibus"
        box_dir = vid + "/box_ibus"

        img_src = cv2.imread(f'{img_dir}/{src_name}.jpg')
        lmks_src = self.__load_label(f'{landmark_dir}/{src_name}.txt', out_shape=(68,2))
        pose_src = self.__load_label(f'{headpose_dir}/{src_name}.txt', out_shape=(-1, 3))

        img_drv = cv2.imread(f'{img_dir}/{drv_name}.jpg')
        lmks_drv = self.__load_label(f'{landmark_dir}/{drv_name}.txt', out_shape=(68,2))
        pose_drv = self.__load_label(f'{headpose_dir}/{drv_name}.txt', out_shape=(-1, 3))

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


        return {"vid_name": self.videos[index], "source":img_src, "driving":img_drv, "lmks_source":{"value": lmks_src}, "lmks_driving":{"value":lmks_drv}, "pose_src":pose_src, "pose_drv":pose_drv}


class VoxDatasetLmks106(VoxDataset):
    def __init__(self, data_dir, transform=False):
        super(VoxDatasetLmks106, self ).__init__(data_dir, transform=transform)

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

    def lmks106_to_lmks98(self, l):
        boundary_and_nose = list(range(0, 55))
        below_nose = list(range(58, 63))
        boundary_left_eye = list(range(66, 74))
        boundary_right_eye = list(range(75, 83))        
        mouth = list(range(84, 104))
        center_left_eye = [104]
        center_right_eye = [105]

        indice = boundary_and_nose + below_nose + boundary_left_eye + boundary_right_eye + mouth + center_left_eye +  center_right_eye
        l = np.array(l)[indice]

        return l

    def lmks98_to_lmks68(self, l):
        LMK98_2_68_MAP = {0: 0,
                        2: 1,
                        4: 2,
                        6: 3,
                        8: 4,
                        10: 5,
                        12: 6,
                        14: 7,
                        16: 8,
                        18: 9,
                        20: 10,
                        22: 11,
                        24: 12,
                        26: 13,
                        28: 14,
                        30: 15,
                        32: 16,
                        33: 17,
                        34: 18,
                        35: 19,
                        36: 20,
                        37: 21,
                        42: 22,
                        43: 23,
                        44: 24,
                        45: 25,
                        46: 26,
                        51: 27,
                        52: 28,
                        53: 29,
                        54: 30,
                        55: 31,
                        56: 32,
                        57: 33,
                        58: 34,
                        59: 35,
                        60: 36,
                        61: 37,
                        63: 38,
                        64: 39,
                        65: 40,
                        67: 41,
                        68: 42,
                        69: 43,
                        71: 44,
                        72: 45,
                        73: 46,
                        75: 47,
                        76: 48,
                        77: 49,
                        78: 50,
                        79: 51,
                        80: 52,
                        81: 53,
                        82: 54,
                        83: 55,
                        84: 56,
                        85: 57,
                        86: 58,
                        87: 59,
                        88: 60,
                        89: 61,
                        90: 62,
                        91: 63,
                        92: 64,
                        93: 65,
                        94: 66,
                        95: 67}

        indice68lmk = np.array(list(LMK98_2_68_MAP.keys()))
        l  = np.array(l)[indice68lmk]

        return l

    def lmks106_to_lmks68(self, l):
        l = self.lmks106_to_lmks98(l)
        l = self.lmks98_to_lmks68(l)
        return l

    def __getitem__(self, index):
        vid = f'{self.data_dir}/{self.videos[index]}'
        
        if not os.path.isdir(f'{vid}/img'):
            return None
            
        frame_ids = os.listdir(f'{vid}/img')


        frame_idx = np.sort(np.random.choice(frame_ids, replace=False, size=2))
        frame_idx = [f[:-4] for f in frame_idx] # 10.png -> 10

        src_name = frame_idx[0]
        drv_name = frame_idx[1]
        img_dir = vid + "/img"
        headpose_dir = vid + "/headpose_ibus"
        landmark_dir = vid + "/landmark106"
        box_dir = vid + "/box_ibus"

        img_src = cv2.imread(f'{img_dir}/{src_name}.jpg')
        lmks_src = self.__load_label(f'{landmark_dir}/{src_name}.txt', out_shape=(106,2))
        lmks_src = self.lmks106_to_lmks68(lmks_src)
        pose_src = self.__load_label(f'{headpose_dir}/{src_name}.txt', out_shape=(-1, 3))

        img_drv = cv2.imread(f'{img_dir}/{drv_name}.jpg')
        lmks_drv = self.__load_label(f'{landmark_dir}/{drv_name}.txt', out_shape=(106,2))
        lmks_drv = self.lmks106_to_lmks68(lmks_drv)
        pose_drv = self.__load_label(f'{headpose_dir}/{drv_name}.txt', out_shape=(-1, 3))

        if lmks_src is  None or lmks_drv is  None:
            return None

        if self.transform:
            scale = self.IMAGE_SIZE / img_src.shape[1]
            lmks_src = lmks_src * scale
            lmks_drv = lmks_drv * scale

            img_src = cv2.resize(img_src, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            img_drv = cv2.resize(img_drv, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            
            transformed = transformerr(image=img_src)
            img_src = transformed["image"]
            transformed = transformerr(image=img_drv)
            img_drv = transformed["image"]


            img_src = img_src / 255.0
            img_src = torch.FloatTensor(img_src).permute(2,0,1)

            img_drv = img_drv / 255.0
            img_drv = torch.FloatTensor(img_drv).permute(2,0,1)

            lmks_src = self.__norm_lmks(lmks_src)
            lmks_src = torch.FloatTensor(lmks_src)
            lmks_drv = self.__norm_lmks(lmks_drv)
            lmks_drv = torch.FloatTensor(lmks_drv)

            # pose_src = torch.FloatTensor(pose_src)
            # pose_drv = torch.FloatTensor(pose_drv)

        # Because when apply horizontaly flip, then lmks should be flip accordingly
        # return {"vid_name": self.videos[index], "source":img_src, "driving":img_drv, "name": self.videos[index]}

        return {"vid_name": self.videos[index], "source":img_src, "driving":img_drv, "lmks_source":{"value": lmks_src}, "lmks_driving":{"value":lmks_drv}, "name": self.videos[index]}


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
    # dset = VoxDataset(data_dir="/home/ubuntu/vuthede/video-preprocessing/processed_vox_dev", transform=True)
    dset = VoxDatasetLmks106(data_dir="/home/ubuntu/vuthede/video-preprocessing/processed_vox_dev", transform=True)
    
    print(f'Len dset :{len(dset)}')
    
    output_debug = "debug_vox_lmks106"
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