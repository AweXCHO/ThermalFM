import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class HADARMultipleScenes():
    NUM_CLASS = 30
    """docstring for HADARSegmentation"""

    def __init__(self, root='/mnt/dqdisk/Code/HADAR/TeXNet/hadar/',
                 split='train', inp_transform=None, target_transform=None,
                 **kwargs):

        root = os.path.expanduser(root)
        self.randflip = kwargs.get('randflip', False) and split == "train"
        fold = kwargs.get('fold', None)

        print("Fold is", fold, "for split", split)

        if fold is None:
            train_ids = [f"L_{i:04d}" for i in range(2, 5)]
            train_ids += [f"R_{i:04d}" for i in range(2, 5)]
            val_ids = ["L_0001", "R_0001"]
            test_ids = ["L_0001", "R_0001"]
            train_exp_ids = train_ids.copy()
            val_exp_ids = ["L_0001", "R_0001"]
            test_exp_ids = ["L_0001", "R_0001"]
        elif fold == 0:
            train_ids = ["L_0002", "L_0003", "L_0004", "L_0005"]
            train_ids += ["R_0002", "R_0003", "R_0004", "R_0005"]
            val_ids = ["L_0001", "R_0001"]
            test_ids = ["L_0001", "R_0001"]
            train_exp_ids = ["L_0002", "L_0003", "L_0004"]
            train_exp_ids += ["R_0002", "R_0003", "R_0004"]
            val_exp_ids = ["L_0001", "R_0001"]
            test_exp_ids = ["L_0001", "R_0001"]
        elif fold == 1:
            train_ids = ["L_0001", "L_0003", "L_0004", "L_0005"]
            train_ids += ["R_0001", "R_0003", "R_0004", "R_0005"]
            val_ids = ["L_0002", "R_0002"]
            test_ids = ["L_0002", "R_0002"]
            train_exp_ids = ["L_0001", "L_0003", "L_0004"]
            train_exp_ids += ["R_0001", "R_0003", "R_0004"]
            val_exp_ids = ["L_0002", "R_0002"]
            test_exp_ids = ["L_0002", "R_0002"]
        elif fold == 2:
            train_ids = ["L_0001", "L_0002", "L_0004", "L_0005"]
            train_ids += ["R_0001", "R_0002", "R_0004", "R_0005"]
            val_ids = ["L_0003", "R_0003"]
            test_ids = ["L_0003", "R_0003"]
            train_exp_ids = ["L_0001", "L_0002", "L_0004"]
            train_exp_ids += ["R_0001", "R_0002", "R_0004"]
            val_exp_ids = ["L_0003", "R_0003"]
            test_exp_ids = ["L_0003", "R_0003"]
        elif fold == 3:
            train_ids = ["L_0001", "L_0002", "L_0003", "L_0005"]
            train_ids += ["R_0001", "R_0002", "R_0003", "R_0005"]
            val_ids = ["L_0004", "R_0004"]
            test_ids = ["L_0004", "R_0004"]
            train_exp_ids = ["L_0001", "L_0002", "L_0003"]
            train_exp_ids += ["R_0001", "R_0002", "R_0003"]
            val_exp_ids = ["L_0004", "R_0004"]
            test_exp_ids = ["L_0004", "R_0004"]
        elif fold == 4:
            train_ids = ["L_0001", "L_0002", "L_0003", "L_0004"]
            train_ids += ["R_0001", "R_0002", "R_0003", "R_0004"]
            val_ids = ["L_0005", "R_0005"]
            test_ids = ["L_0005", "R_0005"]
            train_exp_ids = ["L_0001", "L_0002", "L_0003"]
            train_exp_ids += ["R_0001", "R_0002", "R_0003"]
            val_exp_ids = ["L_0004", "R_0004"]
            test_exp_ids = ["L_0004", "R_0004"]

        if split == 'train':
            ids = train_ids
            exp_ids = train_exp_ids
        elif split == 'val':
            ids = val_ids
            exp_ids = val_exp_ids
        elif split == 'test':
            ids = test_ids
            exp_ids = test_exp_ids

        print('IDs for', split, 'are', ids, 'for synthetic data and', exp_ids, 'for experimental data')

        SUBFOLDERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        SUBFOLDERS = ["Scene"+str(_) for _ in SUBFOLDERS]

        self.S_files = []
        self.S_beta_files = []
        self.T_files = []
        self.e_files = []
        self.v_files = []

        for subfolder in SUBFOLDERS:
            if subfolder == "Scene11":
                ids_ = exp_ids
            else:
                ids_ = ids
            for id in ids_:
                self.S_files.append(os.path.join(root, subfolder, 'HeatCubes',
                                                 f"{id}_heatcube.mat"))
                self.S_beta_files.append(os.path.join(root, subfolder, 'HeatCubes',
                                                 'S_EnvObj_'+f"{id}.npy"))
                self.T_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                                 'tMap', f"tMap_{id}.mat"))
                self.e_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                                 'eMap', f"new_eMap_{id}.npy"))
                self.v_files.append(os.path.join(root, subfolder, 'GroundTruth',
                                                 'vMap', f"vMap_{id}.mat"))

        self.inp_transforms = inp_transform
        self.tgt_transforms = target_transform
        self.split = split

        ######################### Synthetic data ###############################
        self.S_mu = np.array([0.12647585, 0.12525924, 0.12395189, 0.12230065, 0.12088306, 0.11962758,
                              0.11836884, 0.11685297, 0.11524992, 0.11388518, 0.11242859, 0.11083422,
                              0.1090912,  0.10737984, 0.10582539, 0.10439677, 0.10263842, 0.10100006,
                              0.0992386,  0.09752469, 0.09576828, 0.09412399, 0.09233064, 0.09060183,
                              0.08907408, 0.08732026, 0.08569063, 0.08377189, 0.08205311, 0.08037362,
                              0.07875945, 0.07714489, 0.07552012, 0.07388812, 0.07219477, 0.07086218,
                              0.06908296, 0.06754399, 0.06604221, 0.06459464, 0.06316591, 0.06165175,
                              0.0602433,  0.05895745, 0.05754419, 0.05616417, 0.05485069, 0.05351864,
                              0.05223851, 0.05066062, 0.0497363,  0.04859088, 0.04738823, 0.04625365])
        self.S_std = np.array([0.01246481, 0.01251194, 0.0125624,  0.01247964, 0.01251399, 0.01243262,
                               0.0126455,  0.01277499, 0.01247264, 0.01214948, 0.0120328,  0.01196929,
                               0.01211039, 0.01225081, 0.01208897, 0.01186716, 0.01193683, 0.0117601,
                               0.01175319, 0.01168863, 0.01167074, 0.01148603, 0.01150049, 0.01145063,
                               0.0112397,  0.01121394, 0.01108842, 0.01126549, 0.01120692, 0.01110797,
                               0.0109529,  0.01082223, 0.01075425, 0.01073532, 0.01059674, 0.01059848,
                               0.00972673, 0.0094929,  0.00935684, 0.0091823,  0.00900696, 0.00897071,
                               0.00884406, 0.00861178, 0.00857944, 0.00842725, 0.00828631, 0.00812178,
                               0.00806904, 0.00849851, 0.00772755, 0.00772355, 0.00759959, 0.00748127])

        self.T_mu = 15.997467357494212
        self.T_std = 8.544861474951992

        self.S_mu = np.reshape(self.S_mu, (-1, 1, 1))[4:53]
        self.S_std = np.reshape(self.S_std, (-1, 1, 1))[4:53]

        self._load_data()  # Loads data to CPU

        # ---------- 新增：为 val/test 预计算滑动窗口位置 ----------
        self.crop_size = 224
        self.stride = 56

        self.test_positions = []  # list of tuples (img_idx, left, top)
        if self.split != 'train':
            for img_idx, S in enumerate(self.S):
                # S shape: (C, H, W)
                _, H, W = S.shape
                th, tw = self.crop_size, self.crop_size

                xs = list(range(0, max(1, W - tw + 1), self.stride))
                ys = list(range(0, max(1, H - th + 1), self.stride))
                if len(xs) == 0 or xs[-1] != max(0, W - tw):
                    xs.append(max(0, W - tw))
                if len(ys) == 0 or ys[-1] != max(0, H - th):
                    ys.append(max(0, H - th))

                for left in xs:
                    for top in ys:
                        self.test_positions.append((img_idx, left, top))

        # 训练时每张图在一个 epoch 中出现的随机裁剪次数
        self.train_crops_per_image = 1

    def _load_data(self):
        self.S_beta = []
        self.S = []
        self.tMaps = []
        self.eMaps = []
        self.vMaps = []

        for i in self.S_beta_files:
            data = np.load(i)
            data = np.squeeze(data)
            if data.shape[0] == 54:
                data = data[4:53]
            data = torch.from_numpy(data).type(torch.float)
            self.S_beta.append(data)

        for i in self.S_files:
            data = scio.loadmat(i)
            if "S" in data.keys():
                data = data["S"]
            elif "HSI" in data.keys():
                data = data["HSI"]
            else:
                raise ValueError("Known keys not present in heatcubes")
            data = np.transpose(data, (2, 0, 1))
            if data.shape[0] == 54:
                data = data[4:53]
            data = (data - self.S_mu) / self.S_std
            data = torch.from_numpy(data).type(torch.float)
            self.S.append(data)

        for i in self.T_files:
            data = scio.loadmat(i)["tMap"]
            data = (data - self.T_mu) / self.T_std
            data = torch.from_numpy(data).type(torch.float)
            self.tMaps.append(data)

        for i in self.e_files:
            data = np.load(i)
            data = torch.from_numpy(data).type(torch.long)
            self.eMaps.append(data)

        for i in self.v_files:
            data = scio.loadmat(i)["vMap"]
            data = np.transpose(data, (2, 0, 1))
            data = torch.from_numpy(data).type(torch.float)
            self.vMaps.append(data)

        self.num_points = len(self.S_files)

    def __len__(self):
        if self.split == "train":
            return self.num_points * self.train_crops_per_image
        else:
            return len(self.test_positions)

    def __getitem__(self, index_):
        if self.split == 'train':
            img_idx = index_ % self.num_points
            S_beta = self.S_beta[img_idx]
            S = self.S[img_idx]
            tMap = self.tMaps[img_idx]
            eMap = self.eMaps[img_idx]
            vMap = self.vMaps[img_idx]

            th, tw = self.crop_size, self.crop_size
            _, H, W = S.shape
            top = random.randint(0, H - th) if H - th > 0 else 0
            left = random.randint(0, W - tw) if W - tw > 0 else 0

            # apply input transforms to full S if provided
            if self.inp_transforms is not None:
                S = self.inp_transforms(S)

            # crop S, tMap, eMap, vMap using correct top/left order
            S_crop = transforms.functional.resized_crop(S, top, left, th, tw, self.crop_size)

            tMap_u = torch.unsqueeze(tMap, 0)
            tMap_crop = transforms.functional.resized_crop(tMap_u, top, left, th, tw, self.crop_size)
            tMap_crop = torch.squeeze(tMap_crop, 0)

            vMap_crop = transforms.functional.resized_crop(vMap, top, left, th, tw, self.crop_size)

            eMap_u = torch.unsqueeze(eMap, 0)
            eMap_crop = transforms.functional.resized_crop(eMap_u, top, left, th, tw, self.crop_size)
            eMap_crop = torch.squeeze(eMap_crop, 0)

            if self.randflip:
                flip = torch.rand(1) > 0.5
                if flip:
                    S_crop = TF.hflip(S_crop)
                    tMap_crop = TF.hflip(tMap_crop)
                    eMap_crop = TF.hflip(eMap_crop)
                    vMap_crop = TF.hflip(vMap_crop)

            S_crop = S_crop.to(torch.float32)
            tMap_crop = tMap_crop.to(torch.float32)
            vMap_crop = vMap_crop.to(torch.float32)
            eMap_crop = eMap_crop.to(torch.long)

            return S_beta, S_crop, (tMap_crop, eMap_crop, vMap_crop)

        else:
            img_idx, left, top = self.test_positions[index_]
            S_beta = self.S_beta[img_idx]
            S = self.S[img_idx]
            tMap = self.tMaps[img_idx]
            eMap = self.eMaps[img_idx]
            vMap = self.vMaps[img_idx]

            th, tw = self.crop_size, self.crop_size

            if self.inp_transforms is not None:
                S = self.inp_transforms(S)

            S_crop = transforms.functional.resized_crop(S, top, left, th, tw, self.crop_size)

            tMap_u = torch.unsqueeze(tMap, 0)
            tMap_crop = transforms.functional.resized_crop(tMap_u, top, left, th, tw, self.crop_size)
            tMap_crop = torch.squeeze(tMap_crop, 0)

            vMap_crop = transforms.functional.resized_crop(vMap, top, left, th, tw, self.crop_size)

            eMap_u = torch.unsqueeze(eMap, 0)
            eMap_crop = transforms.functional.resized_crop(eMap_u, top, left, th, tw, self.crop_size)
            eMap_crop = torch.squeeze(eMap_crop, 0)

            S_crop = S_crop.to(torch.float32)
            tMap_crop = tMap_crop.to(torch.float32)
            vMap_crop = vMap_crop.to(torch.float32)
            eMap_crop = eMap_crop.to(torch.long)

            return S_beta, S_crop, (tMap_crop, eMap_crop, vMap_crop)


class HADARMultipleScenesLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.data_dir = args.data_dir
        self.args = args
        self.num_workers = args.workers
        self.dataset_name = args.dataset
        self.dataset_type = HADARMultipleScenes
        self.randerase = args.randerase
        self.randflip = args.randerase
        self.eval_on_train = args.eval_on_train
        self.fold = args.fold
        self.eval_only = args.eval

    def setup(self, stage=None):

        train_inp_transform_list = []
        if self.randerase and False:
            train_inp_transform_list.extend([transforms.RandomErasing(p=0.5, scale=(0.1, 0.5))])

        if len(train_inp_transform_list) > 0:
            train_inp_transform = transforms.Compose(train_inp_transform_list)
        else:
            train_inp_transform = None

        train_tgt_transform = None

        if not self.eval_only:
            print(f"** Loading training dataset....")
            self.train_data = self.dataset_type(root=self.data_dir,
                                                split='train',
                                                inp_transform=train_inp_transform,
                                                target_transform=train_tgt_transform,
                                                randflip=self.randflip,
                                                fold=self.fold)
            self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True, drop_last=False,
                                persistent_workers=True)

        print(f"** Loading validation dataset....")
        self.val_data = self.dataset_type(root=self.data_dir,
                                          split='val',
                                          inp_transform=None,
                                          target_transform=None,
                                          fold=self.fold)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=True)

        # 如果需要 test loader，可按需打开（与 val 一致）
        # self.test_data = self.dataset_type(root=self.data_dir,
        #                                   split='test',
        #                                   inp_transform=None,
        #                                   target_transform=None,
        #                                   fold=self.fold)
        # self.test_loader = DataLoader(self.test_data, batch_size=1,
        #                     num_workers=self.num_workers, pin_memory=True, drop_last=False,
        #                     persistent_workers=True)

    def train_dataloader(self):
        
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return getattr(self, "test_loader", None)
