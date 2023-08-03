import torch
from pycocotools import coco
import copy
import os
import numpy as np
from imageio import imread
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as tff
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import augument.trans as mt


def filt_small_instance(coco_item, pixthreshold=4000, imgNthreshold=5):
    list_dict = coco_item.catToImgs
    for catid in list_dict:
        list_dict[catid] = list(set(list_dict[catid]))
    new_dict = copy.deepcopy(list_dict)
    for catid in list_dict:
        imgids = list_dict[catid]
        for n in range(len(imgids)):
            imgid = imgids[n]
            anns = coco_item.imgToAnns[imgid]
            has_large_instance = False
            for ann in anns:
                if (ann['category_id'] == catid) and (ann['iscrowd'] == 0) and (ann['area'] > pixthreshold):
                    has_large_instance = True
            if has_large_instance is False:
                new_dict[catid].remove(imgid)
        imgN = len(new_dict[catid])
        if imgN < imgNthreshold:
            new_dict.pop(catid)
            print('catid:%d  remain %d images, delet it!' % (catid, imgN))
        else:
            print('catid:%d  remain %d images' % (catid, imgN))
    print('remain  %d  categories' % len(new_dict))
    np.save('./utils/new_cat2imgid_dict%d.npy' % pixthreshold, new_dict)
    return new_dict

# no data argumentation
class CoCoDataset(torch.utils.data.Dataset):
    """coco 数据集"""
    def __init__(self, data_path, npy_path, anf_path):
        # 检测路径
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        assert os.path.exists(npy_path), "Npy path '{}' not found".format(npy_path)
        assert os.path.exists(anf_path), "Anf path '{}' not found".format(anf_path)
        self.data_path = data_path
        self.list_dict = np.load(npy_path, allow_pickle=True).item()
        self.anf_path = anf_path
        self.catid2label = dict()
        self.coco = coco.COCO(annotation_file=anf_path)
        for i, cat_id in enumerate(self.list_dict):
            self.catid2label[cat_id] = i
        self.idx2catid = [i for i in self.list_dict]
        self.img_size = 224
        self.group_size = 5                 # 一次取一个group的数据

        self.trans = mt.Compose([
            mt.Resize((self.img_size, self.img_size)),
            mt.ToTensor(),
            mt.Normalize()
        ])

    def __getitem__(self, idx):
        """
        拿数据
        :param idx:  group id
        :return:
        """
        cat_id = self.idx2catid[idx]
        img_ids = random.sample(self.list_dict[cat_id], self.group_size)

        # 处理每个group的第一张图片
        cls_labels = torch.zeros(78)
        anns = self.coco.imgToAnns[img_ids[0]]
        cat_ids_mix = set()
        for ann in anns:
            if (ann['iscrowd'] == 0) and (ann['area'] > 4000):  # 如果这个标注信息满足这样的需求
                cat_ids_mix.add(ann['category_id'])     # 就把他加入 co_cat_mix

        # 对所有图片都标注的地方求交集，这里选择了跟源代码不同的写法，用set完成了
        for img_id in img_ids[1:]:
            cat_ids_tmp = set()
            anns = self.coco.imgToAnns[img_id]   # 获取当前图片的标注信息
            for ann in anns:
                if (ann['iscrowd'] == 0) and (ann['area'] > 4000):
                    cat_ids_tmp.add(ann['category_id'])
            cat_ids_mix = cat_ids_mix & cat_ids_tmp     # 求交集

        for co_cat_id in cat_ids_mix:
            cls_labels[self.catid2label[co_cat_id]] = 1

        imgs = torch.zeros((0, 3, self.img_size, self.img_size))
        mask_labels = torch.zeros((0, self.img_size, self.img_size))
        for imgid in img_ids:
            im_path = self.data_path + '%012d.jpg' % imgid
            img = Image.open(im_path)

            anns = self.coco.imgToAnns[imgid]
            mask = None
            for ann in anns:
                if ann['category_id'] in cat_ids_mix:
                    if mask is None:
                        mask = self.coco.annToMask(ann)
                    else:
                        mask = mask + self.coco.annToMask(ann)
            mask[mask > 0] = 255
            mask = Image.fromarray(mask)
            img, mask = self.trans(img, mask)
            img = img.unsqueeze(0)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            mask_labels = torch.concat((mask_labels, mask))
            imgs = torch.concat((imgs, img))

        return imgs, cls_labels, mask_labels

    def __len__(self):
        return len(self.idx2catid)


def img_normalize(image):
    if len(image.shape) == 2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel, channel, channel], axis=2)
    else:
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))) \
                / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image

def generate_batch(batch_data):
    imgs_list = []
    cls_labels_list = []
    mask_labels_list = []
    for bd in batch_data:
        imgs_list.append(bd[0])
        cls_labels_list.append(bd[1])
        mask_labels_list.append(bd[2])
    return torch.concat(imgs_list), torch.stack(cls_labels_list), torch.concat(mask_labels_list)

davis_fbms = ['bear', 'bear01', 'bear02', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'cars2', 'cars3',
              'cars6', 'cars7', 'cars8', 'cars9', 'cats02', 'cats04', 'cats05', 'cats07', 'dance-jump', 'dog-agility',
              'drift-turn', 'ducks01', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 'horses01',
              'horses03', 'horses06', 'kite-walk', 'lion02', 'lucia', 'mallard-fly', 'mallard-water', 'marple1',
              'marple10', 'marple11', 'marple13', 'marple3', 'marple5', 'marple8', 'meerkats01', 'motocross-bumps',
              'motorbike', 'paragliding', 'people04', 'people05', 'rabbits01', 'rabbits05', 'rhino', 'rollerblade',
              'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']


class VideoDataset(Dataset):
    def __init__(self, dir_, epochs, size=224, group=5, use_flow=False):

        self.img_list = []
        self.label_list = []
        self.flow_list = []

        self.group = group

        dir_img = os.path.join(dir_, 'image')
        dir_gt = os.path.join(dir_, 'groundtruth')
        dir_flow = os.path.join(dir_, 'flow')
        self.dir_list = sorted(os.listdir(dir_img))
        self.leng = 0
        for i in range(len(self.dir_list)):
            ok = 0
            if self.dir_list[i] in davis_fbms:
                ok = 1
            if ok == 0:
                continue
            tmp_list = []
            cur_dir = sorted(os.listdir(os.path.join(dir_img, self.dir_list[i])))
            for j in range(len(cur_dir)):
                tmp_list.append(os.path.join(dir_img, self.dir_list[i], cur_dir[j]))
            self.leng += len(tmp_list)
            self.img_list.append(tmp_list)

            tmp_list = []
            cur_dir = sorted(os.listdir(os.path.join(dir_gt, self.dir_list[i])))
            for j in range(len(cur_dir)):
                tmp_list.append(os.path.join(dir_gt, self.dir_list[i], cur_dir[j]))
            self.label_list.append(tmp_list)

        self.img_size = 224
        self.dataset_len = epochs
        self.use_flow = use_flow
        self.dir_ = dir_

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        rd = np.random.randint(0, len(self.img_list))
        rd2 = np.random.permutation(len(self.img_list[rd]))
        cur_img = []
        cur_flow = []
        cur_gt = []
        for i in range(self.group):
            cur_img.append(self.img_list[rd][rd2[i % len(self.img_list[rd])]])
            cur_flow.append(
                os.path.join(self.dir_, 'flow', os.path.split(self.img_list[rd][rd2[i % len(self.img_list[rd])]])[1]))
            cur_gt.append(self.label_list[rd][rd2[i % len(self.img_list[rd])]])

        group_img = [] # img
        group_flow = []
        group_gt = []
        for i in range(self.group):
            tmp_img = imread(cur_img[i])

            tmp_img = torch.from_numpy(img_normalize(tmp_img.astype(np.float32) / 255.0))
            tmp_img = F.interpolate(tmp_img.unsqueeze(0).permute(0, 3, 1, 2), size=(self.img_size, self.img_size))
            group_img.append(tmp_img)

            tmp_gt = np.array(Image.open(cur_gt[i]).convert('L')) # gray img
            tmp_gt = torch.from_numpy(tmp_gt.astype(np.float32) / 255.0)
            tmp_gt = F.interpolate(tmp_gt.view(1, tmp_gt.shape[0], tmp_gt.shape[1], 1).permute(0, 3, 1, 2),
                                   size=(self.img_size, self.img_size)).squeeze()  # up/down sampling for tensors
            tmp_gt = tmp_gt.view(1, tmp_gt.shape[0], tmp_gt.shape[1])
            group_gt.append(tmp_gt)
            if self.use_flow == True:
                tmp_flow = imread(cur_flow[i])
                tmp_flow = torch.from_numpy(img_normalize(tmp_flow.astype(np.float32) / 255.0))
                tmp_flow = F.interpolate(tmp_flow.unsqueeze(0).permute(0, 3, 1, 2), size=(self.img_size, self.img_size))
                group_flow.append(tmp_flow)

        group_img = (torch.cat(group_img, 0))
        if self.use_flow == True:
            group_flow = torch.cat(group_flow, 0)
        group_gt = (torch.cat(group_gt, 0))
        if self.use_flow == True:
            return group_img, group_flow, group_gt
        else:
            return group_img, group_gt

if __name__ == '__main__':
    npy = './utils/new_cat2imgid_dict4000.npy'
    dp = '/root/autodl-tmp/coco2017/train2017/'
    pics_fp = dp + os.listdir(dp)[5]

    anp = '/root/autodl-tmp/coco2017/annotations/instances_train2017.json'
