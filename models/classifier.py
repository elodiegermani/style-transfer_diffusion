import torch.nn as nn

class Regressor3D(nn.Module):

    def __init__(self):
        super(Regressor3D, self).__init__()

        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1, stride=2)
        self.norm1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2)
        self.norm2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2)
        self.norm3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=2)
        self.norm4 = nn.BatchNorm3d(256)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1, stride=2)
        self.norm5 = nn.BatchNorm3d(512)
        self.flatten = nn.Flatten()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.lrelu(x)
        x = self.flatten(x)
        
        return x
    
class Classifier3D(nn.Module):

    def __init__(self, n_class):
        super(Classifier3D, self).__init__()

        self.lrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1, stride=2)
        self.norm1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2)
        self.norm2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2)
        self.norm3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=2)
        self.norm4 = nn.BatchNorm3d(256)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1, stride=2)
        self.norm5 = nn.BatchNorm3d(512)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, n_class)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.lrelu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        
        return x

    def predict(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.lrelu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        
        return x