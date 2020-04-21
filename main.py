from torchvision import transforms
from preprocessing import preprocessing
from make_dataset import Mydataset



BATCH_SIZE = None
SHUFFLE = True
NUM_WORKERS = 4 #multithreading



transform = transforms.Compose([transforms.ToTensor()])
dataset = Mydataset(sources,targets,trnasform)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)