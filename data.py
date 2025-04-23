from torchvision import transforms as tfs
from torchvision  import datasets as dst
from torch.utils.data import random_split as random
data_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Resize((224, 224)),
    tfs.Grayscale()
])
data = dst.ImageFolder(r'/data', transform = data_tfs)                   # Create dataset.
len_data_train = int(0.7*len(data))                                      # Define len train and test data (default: train 70%, test 30%).
len_data_test = int(0.3*len(data))                                       # If you want to change len train and test you should change
                                                                         # "int(0.7*len(data))" to your values.
                                                                         # For ex.,  len_data_train = 10000.     

train_data = random(data, [len_data_train, len(data)-len_data_train])[0] # Split dataset to train and test with random sequence 
test_data = random(data, [len_data_test, len(data)-len_data_test])[0]