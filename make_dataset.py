#len(dataset) 에서 호출되는 __len__ 은 데이터셋의 크기를 리턴해야합니다.
#dataset[i] 에서 호출되는 __getitem__ 은 i번째 샘플을 찾는데 사용됩니다.

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



class Mydataset(Dataset):
    def __init__(self, source, target, transform=None, root_dir=None):
        self.source = source
        self.root_dir = root_dir
        self.transform = transform
        self.target = target

    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        """
        data_name = os.path.join(self.root_dir, self.data_list[idx])
        data = io.imread(data_name)
        """

        sample = {'source': self.source[idx], 'target': self.target[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

