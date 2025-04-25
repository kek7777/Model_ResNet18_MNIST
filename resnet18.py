from torch import nn 
#building ResNet 18  (it is all architecture) 
class ResNet18 (nn.Module):
    def __init__(self, Residual_block, num_classes=10):
         super(ResNet18, self).__init__()
         self.in_block = 64                                               # 1 input image channel, 64 output channels, 7x7 square conv kernel
         self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias=False) #  на выходе получаем 64 шт. 112*112

         self.bn1 = nn.BatchNorm2d(64)
         self.relu = nn.ReLU(inplace=True)
         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #  на выходе получаем 64 шт. 55*55
         
         self.layer1 = self.make_layer(Residual_block, 64, 2, stride=1)    #  создание 4 остаточных групп (2 - количество блоков в группе)
         self.layer2 = self.make_layer(Residual_block, 128, 2, stride=2)
         self.layer3 = self.make_layer(Residual_block, 256, 2, stride=2)
         self.layer4 = self.make_layer(Residual_block, 512, 2, stride=2)
        
         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))      #  выход (1,1)
         self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_block, out_block, stride))
            self.in_block = out_block
        return nn.Sequential(*layers)                             # выводит не список, а значения элементов списка
  
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)             # расчет функции потерь; view(-1) преобразует torch.Size([128, 1])  в torch.Size([128])
        out = self.fc(out)
        # print (out.shape)
        # exit
        return out
    