from torch import nn 

#building residual block (num = 8)

class Residual_block (nn.Module):
    def __init__(self, in_block, out_block, stride=1):
         super(Residual_block, self).__init__()
         self.conv1 = nn.Conv2d(in_block, out_block, kernel_size=3, stride=stride, padding=1, bias=False)
         self.bn1 = nn.BatchNorm2d(out_block)
         self.relu = nn.ReLU(inplace=True)
         self.conv2 = nn.Conv2d(out_block, out_block, kernel_size=3, stride=1, padding=1, bias=False)
         self.bn2 = nn.BatchNorm2d(out_block)

         self.skip_connect = nn.Sequential()
         if stride != 1 or in_block != out_block:
              self.skip_connect = nn.Sequential(
                nn.Conv2d(in_block, out_block, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_block)
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.skip_connect(x)
        out = self.relu(out)
        return out
          
        