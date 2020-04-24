#len(dataset) 에서 호출되는 __len__ 은 데이터셋의 크기를 리턴해야합니다.
#dataset[i] 에서 호출되는 __getitem__ 은 i번째 샘플을 찾는데 사용됩니다.
import torch
from torch.utils.data import Dataset

class trans (object):
    def __init__(self):
        self.sample = {}
    
    def transform(self, sample):
        self.sample['source'] = torch.FloatTensor(sample['source'])
        self.sample['target'] = torch.FloatTensor(sample['target'])
        return self.sample
        

class Mydataset(Dataset):
    def __init__(self, sources, targets, transform=None, root_dir=None):
        self.sources = sources
        self.root_dir = root_dir
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'source': self.sources[idx], 'target': self.targets[idx]}

        if self.transform:
            sample = self.transform.transform(sample)

        return sample

