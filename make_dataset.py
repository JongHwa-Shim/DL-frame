#len(dataset) 에서 호출되는 __len__ 은 데이터셋의 크기를 리턴
#dataset[i] 에서 호출되는 __getitem__ 은 i번째 샘플을 찾는데 사용
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class self_transform (object):
    def __init__(self, *kwargs):
        self.toPIL = transforms.ToPILImage()
        self.transforms = [transforms for transforms in kwargs]

    def __call__(self, sample):
        real = sample['real']
        condition = sample['condition']

        # real image processing
        real = torch.FloatTensor(real)
        real = real.view(1,1,-1)
        real = self.toPIL(real)
        if self.transforms:
            for transform in self.transforms:
                real = transform(real)
        real = real.view(-1)

        # condition processing
        condition = torch.FloatTensor(condition)
        

        sample['real'] = real
        sample['condition'] = condition
        return sample
        

class Mydataset(Dataset):
    def __init__(self, sources, targets, transform=None, root_dir=None):
        self.sources = sources
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'real': self.sources[idx], 'condition': self.targets[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

