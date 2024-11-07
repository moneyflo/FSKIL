import os
import random
from glob import glob

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

SR = 16000

class GSC2(Dataset):

    def __init__(self, root='/home/seok/localDB', train=True,
                 index_path=None, index=None, base_sess=False):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.base_sess = base_sess
        self._pre_operate(self.root)

        noise_dir = '/home/seok/workspace/fskil/FSKIL/data/datasets/GSC2/_background_noise_'
        self.preprocess = Preprocess(noise_loc=noise_dir)
        
        if train:
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        audio_file = os.path.join(root, 'GSC2/keywords.txt')
        split_file = os.path.join(root, 'GSC2/train_test_split.txt')
        class_file = os.path.join(root, 'GSC2/keyword_class_labels.txt')
        id2audio = self.list2dict(self.text_read(audio_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train audios; 0: test audios
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                audio_path = os.path.join(root, 'GSC2/keywords', id2audio[k])
                self.data.append(audio_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[audio_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                audio_path = os.path.join(root, 'GSC2/keywords', id2audio[k])
                self.data.append(audio_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[audio_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        
        sample, _ = torchaudio.load(path)
        sample = self.preprocess(sample, is_train=self.train, base_sess=self.base_sess)    
        
        return sample, targets


class Preprocess:
    def __init__(
        self,
        noise_loc,
        hop_length=160,
        win_length=480,
        n_fft=512,
        n_mels=40,
        sample_rate=SR,
    ):
        if noise_loc is None:
            self.background_noise = []
        else:
            self.background_noise = [
                torchaudio.load(file_name)[0] for file_name in glob(noise_loc + "/*.wav")
            ]
            assert len(self.background_noise) != 0
        self.feature = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        )
        
        self.sample_len = sample_rate
        
    def _pad_audio(self, x):
        """Zero pad audio totensure 1 second length!!!"""
        pad_len = SR - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, torch.zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            raise ValueError("no sample exceed 1sec in GSC.")
        return x

    def _apply_augmentation(self, x, noise_prob):
        """Apply audio augmentation."""
        for idx in range(x.shape[0]):
            if (random.random() > noise_prob):
                continue
            noise_amp = np.random.uniform(0, 1)
            noise = random.choice(self.background_noise)
            sample_loc = random.randint(0, noise.shape[-1] - self.sample_len)
            noise = noise_amp * noise[:, sample_loc : sample_loc + SR]
        
            x_shift = int(np.random.uniform(-0.1, 0.1) * SR)
            zero_padding = torch.zeros(np.abs(x_shift))
            
            if x_shift < 0:
                temp_x = torch.cat([zero_padding, x[idx, :x_shift]], dim=-1)
            else:
                temp_x = torch.cat([x[idx, x_shift:], zero_padding], dim=-1)
            x[idx] = torch.clamp(temp_x + noise, -1.0, 1.0)
                
        return x
    
    def __call__(self, x, noise_prob=0.8, is_train=True, base_sess=False):
        x = self._pad_audio(x)
        if is_train and base_sess:
            x = self._apply_augmentation(x, noise_prob)
        
        return (self.feature(x) + 1e-6).log()


if __name__ == '__main__':
    txt_path = "../../data/index_list/gscv2/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 20
    class_index = np.arange(base_class)
    dataroot = '/home/seok/localDB'
    batch_size_base = 10
    trainset = GSC2(root=dataroot, train=False,  index=class_index,
                      base_sess=True)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=1,
                                              pin_memory=True)

    for x, y in trainloader:
        print(x, y)
        break
    import pdb; pdb.set_trace()