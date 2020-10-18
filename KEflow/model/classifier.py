import torch
from torch import nn

class BasicCNN(nn.Module):
    def __init__(self, nc=1, im_size=32, n_filter=16, n_class=10 ):
        super(BasicCNN, self).__init__()
        self.layer_stack = torch.nn.Sequential( 
            nn.Conv2d(nc, n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),

            nn.Conv2d(n_filter, n_filter, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),


            # B x n_filter x 16 x 16
            nn.Conv2d(n_filter, 2*n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),

            nn.Conv2d(2*n_filter, 2*n_filter, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),          

            # B x 2*n_filter x 8 x 8
            nn.Conv2d(2*n_filter, 2*n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),

            nn.Conv2d(2*n_filter, 2*n_filter, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),   


            # B x 4*n_filter x 4 x 4
            # fc layer in filter format
            nn.Conv2d(2*n_filter, 4*n_filter, 4),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*n_filter)        
        )
        self.fc = nn.Linear(4*n_filter, n_class)

    def forward(self, x):
        x = self.layer_stack(x)
        logit = self.fc(x.squeeze())
        return logit


class LeNet5(nn.Module):
    """
    Class for a Lenet5 classifier.
    """

    def __init__(self, nc=1, im_size=32, n_filter=16, n_class=10 ):
        """
        Class initializer.
        """
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.max_pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.max_pool2 = nn.MaxPool2d((2, 2), 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Forward propagation.
        """
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.max_pool2(out)
        out = self.conv3(out)
        out = torch.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out

 

        
if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = LeNet5(1, 32, 16, 10)
    teacher = count_parameters(model)
    model.compress(2)
    student = count_parameters(model)
    print(teacher, student)
    print(teacher/student)
