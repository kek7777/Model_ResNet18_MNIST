from torchvision import transforms as tfs
from torchvision  import datasets as dst
data_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Resize((224, 224)),
    tfs.Grayscale()
])
train_data = dst.ImageFolder(r'data\Train_data', transform = data_tfs)
test_data =  dst.ImageFolder(r'data\Test_data', transform = data_tfs)