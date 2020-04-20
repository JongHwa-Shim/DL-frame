#len(dataset) 에서 호출되는 __len__ 은 데이터셋의 크기를 리턴해야합니다.
#dataset[i] 에서 호출되는 __getitem__ 은 i번째 샘플을 찾는데 사용됩니다.

from preprocessing import *
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

class Mydataset(Dataset):
    def __init__(self, data, label, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        """
        data_name = os.path.join(self.root_dir, self.data_list[idx])
        data = io.imread(data_name)
        """

        sample = {'data': self.data[idx], 'label': self.label[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
