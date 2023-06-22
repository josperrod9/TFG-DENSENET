import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

class HandDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        self.img_size = 224
        self.joints = 21  # 21 heat maps
        self.label_size = 2  # (x, y)
        self.mode = mode

        self.data_root = data_root
        self.img_names = json.load(open(os.path.join(self.data_root, 'partitions.json')))[mode]
        self.all_labels = json.load(open(os.path.join(self.data_root, 'labels.json')))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]  # '00000001.jpg'

        # ********************** get image **********************
        im = Image.open(os.path.join(self.data_root, 'imgs', img_name))
        w, h = im.size
        im = im.resize((self.img_size, self.img_size))

        image = transforms.ToTensor()(im)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)

        # ******************** get label  **********************
        img_label = self.all_labels[img_name]  # origin label list  21 * 2

        label = np.asarray(img_label)  # 21 * 2
        label[:, 0] = label[:, 0] / w
        label[:, 1] = label[:, 1] / h
        return image, label, img_name, w, h


# test case
if __name__ == "__main__":
    data_root = 'data_sample/Panoptic'

    print('Dataset ===========>')
    data = HandDataset(data_root=data_root, mode='train')
   #for i in range(len(data)):
    image, label, img_name, w, h = data[0]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(torch.__version__)
    print(cuda)
    torch.cuda.current_device()
    #    # ***************** draw Limb map *****************
    #    print(image.shape, label, img_name, w, h)





