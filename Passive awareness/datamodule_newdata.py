import os
import numpy as np
import scipy.io as scio
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class HADARMultipleScenes():
    NUM_CLASS = 30

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

        SUBFOLDERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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

        self.num_points = len(self.S_files)

        # Synthetic normalization stats (unchanged)
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

        self._load_data()

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

    def __len__(self):
        # 验证/测试时每个样本返回 5 个裁剪（中心 + 四角）
        if self.split == "train":
            return self.num_points
        else:
            return 5 * self.num_points

    def __getitem__(self, index_):
        # 训练：随机裁剪；验证/测试：5 个固定裁剪（size_idx 决定位置）
        if self.split == 'train':
            index = index_
            size_idx = 0
        else:
            index = index_ // 5
            size_idx = index_ % 5

        S_beta = self.S_beta[index]
        S = self.S[index]  # shape: C x H x W
        tMap = self.tMaps[index]  # shape: H x W
        eMap = self.eMaps[index]  # shape: H x W (long)
        vMap = self.vMaps[index]  # shape: C_v x H x W

        if self.inp_transforms is not None:
            S = self.inp_transforms(S)

        if self.tgt_transforms is not None:
            tMap = self.tgt_transforms(tMap)
            eMap = self.tgt_transforms(eMap)
            vMap = self.tgt_transforms(vMap)

        # 目标裁剪尺寸
        crop_h, crop_w = 448, 448

        # 获取当前图像尺寸（注意 S 的 shape 是 C x H x W）
        _, H, W = S.shape

        # 如果图像比裁剪尺寸小，则先用中心填充到至少裁剪尺寸
        pad_h = max(0, crop_h - H)
        pad_w = max(0, crop_w - W)
        if pad_h > 0 or pad_w > 0:
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            S = TF.pad(S, (left, top, right, bottom), fill=0)
            tMap = TF.pad(tMap.unsqueeze(0), (left, top, right, bottom), fill=0).squeeze(0)
            eMap = TF.pad(eMap.unsqueeze(0).float(), (left, top, right, bottom), fill=0).squeeze(0).long()
            vMap = TF.pad(vMap, (left, top, right, bottom), fill=0)
            _, H, W = S.shape

        if self.split == 'train':
            # 随机裁剪（训练）
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
        else:
            # 验证/测试：5 个固定裁剪
            # size_idx mapping:
            # 0 -> center
            # 1 -> top-left
            # 2 -> top-right
            # 3 -> bottom-left
            # 4 -> bottom-right
            if size_idx == 0:
                top = (H - crop_h) // 2
                left = (W - crop_w) // 2
            elif size_idx == 1:
                top = 0
                left = 0
            elif size_idx == 2:
                top = 0
                left = max(0, W - crop_w)
            elif size_idx == 3:
                top = max(0, H - crop_h)
                left = 0
            elif size_idx == 4:
                top = max(0, H - crop_h)
                left = max(0, W - crop_w)
            else:
                # fallback to center
                top = (H - crop_h) // 2
                left = (W - crop_w) // 2

        # 裁剪 S (C x H x W)
        S = transforms.functional.crop(S, top, left, crop_h, crop_w)

        # tMap 和 eMap 是 H x W，需要先加 channel 维度再裁剪，最后去掉
        tMap = tMap.unsqueeze(0)
        tMap = transforms.functional.crop(tMap, top, left, crop_h, crop_w)
        tMap = tMap.squeeze(0)

        eMap = eMap.unsqueeze(0).float()
        eMap = transforms.functional.crop(eMap, top, left, crop_h, crop_w)
        eMap = eMap.squeeze(0).long()

        # vMap 可能是 C_v x H x W
        vMap = transforms.functional.crop(vMap, top, left, crop_h, crop_w)

        # 类型确保
        S = S.to(torch.float32)
        tMap = tMap.to(torch.float32)
        vMap = vMap.to(torch.float32)
        eMap = eMap.to(torch.long)

        # 随机水平翻转（仅训练时启用）
        if self.split == 'train' and self.randflip:
            flip = torch.rand(1) > 0.5
            if flip:
                S = TF.hflip(S)
                tMap = TF.hflip(tMap)
                eMap = TF.hflip(eMap)
                vMap = TF.hflip(vMap)

        return S_beta, S, (tMap, eMap, vMap)


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
                                num_workers=self.num_workers, pin_memory=True, drop_last=True,
                                persistent_workers=True)

        print(f"** Loading validation dataset....")
        self.val_data = self.dataset_type(root=self.data_dir,
                                          split='val',
                                          inp_transform=None,
                                          target_transform=None,
                                          fold=self.fold)
        # 验证时我们返回每个裁剪作为独立样本（batch_size=1），上层评估代码需要把同一原图的 5 个裁剪聚合（index // 5 相同）
        self.val_loader = DataLoader(self.val_data, batch_size=10,
                            num_workers=self.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=True)

        # 如果需要测试集，也可以按同样方式加载（5-crop）
        print(f"** Loading testing dataset....")
        self.test_data = self.dataset_type(root=self.data_dir,
                                          split='test',
                                          inp_transform=None,
                                          target_transform=None,
                                          fold=self.fold)
        self.test_loader = DataLoader(self.test_data, batch_size=10,
                            num_workers=self.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=True)

    def train_dataloader(self):
        
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
