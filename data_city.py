import os
import torch
import numpy as np
import cv2
from torch.utils import data
from vis_task_3_utils import recursive_glob
from vis_task_3_transforms import Compose, RandomHorizontalFlip, RandomRotate, ToTensor,Normalize

class cityscapesLoader(data.Dataset):

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        img_norm=True,
        version="cityscapes",
        test_mode=False,
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 0
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))
        
        self.tf = Compose(                                                   
            [                                                                
                # add more trasnformations as you see fit
                ToTensor(),
                RandomRotate(90),                                                         
                RandomHorizontalFlip(0.3),                                                                                     
                Normalize(mean=[0.28689554, 0.32513303, 0.28389177], std=[0.18696375, 0.19017339, 0.18720214])
            ]                                                                
        )
        self.tf_test = Compose(                                                   
            [                                                                
                # add more trasnformations as you see fit
                ToTensor(),                                                                                   
                Normalize(mean=[0.28689554, 0.32513303, 0.28389177], std=[0.18696375, 0.19017339, 0.18720214])
            ]                                                                
        )

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        
        img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.uint8)
 
 
        lbl = cv2.imread(lbl_path,0)  
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        
        img = cv2.resize(img, dsize=(self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        lbl = cv2.resize(lbl, dsize=(self.img_size[0], self.img_size[1])) 
        
        if self.is_transform:
            img, lbl = self.tf(img, lbl)
        if self.split == 'train':
            lbl = lbl.squeeze()*255
        if self.split != 'train':
            img, lbl = self.tf_test(img, lbl)
            lbl = lbl.squeeze()*255
        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask           