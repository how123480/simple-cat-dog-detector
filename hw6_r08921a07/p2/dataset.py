import os,glob,zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), 
    transforms.ToTensor()])

transform_test = transforms.Compose([
    transforms.ToTensor()])

class Dog_Cat_Dataset(Dataset):
    
    def __init__(self, path='./Cat_Dog_data_small/train', img_size=112, transform = transform_train):
        # download dataset if not exist
        if not os.path.exists(path):
            from utils import download_file_from_google_drive 

            print('\t[Info] Downloading dataset', end='\r')
            download_file_from_google_drive('17WlQZc_5qu61BFjVVaEhMp7XDwYIU5i_', 'cat_dog.zip')
            with zipfile.ZipFile("cat_dog.zip","r") as zip_ref:
                zip_ref.extractall("./")
            os.system('rm cat_dog.zip')
            print('\t[Info] Complete, Download cat dog dataset')
            
        self.img_size = img_size
        self.transform = transform
        cat_imgs = glob.glob(os.path.join(path,'cat/*.jpg'))
        dog_imgs = glob.glob(os.path.join(path,'dog/*.jpg'))
        
        self.data = list(cat_imgs)+list(dog_imgs)
        self.label = [0 for i in range(len(cat_imgs))] + \
                      [1 for i in range(len(dog_imgs))]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        img = Image.open(path).resize((self.img_size,self.img_size))
        img = self.transform(img)
        
        return torch.tensor(img).float() ,\
                torch.tensor(self.label[idx]).long()
    
class Dog_Cat_Dataset_Test(Dataset):
    
    def __init__(self, path, img_size=112, transform=transform_test):

        self.img_size = img_size
        self.transform = transform
        self.path = path
        self.data = list(glob.glob(os.path.join(path, '*.jpg')))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        img = Image.open(path).resize((self.img_size,self.img_size))
        img = self.transform(img)
        
        return torch.tensor(img).float() ,\
                self.data[idx].replace(self.path,'').replace('/','')
    
