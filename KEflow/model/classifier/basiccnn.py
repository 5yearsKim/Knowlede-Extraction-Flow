import torch
from torch import nn

class BasicCNN(nn.Module):
    def __init__(self, nc=1, n_class=10 ):
        super(BasicCNN, self).__init__()
        n_filter = 16
        self.layer_stack = torch.nn.Sequential( 
            nn.Conv2d(nc, n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),

            nn.Conv2d(n_filter, n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),
            
            nn.MaxPool2d((2, 2), 2)

            # B x n_filter x 16 x 16
            nn.Conv2d(n_filter, 2*n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),

            nn.Conv2d(2*n_filter, 2*n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),          
            
            nn.MaxPool2d((2, 2), 2)

            # B x 2*n_filter x 8 x 8
            nn.Conv2d(2*n_filter, 2*n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),

            nn.Conv2d(2*n_filter, 2*n_filter, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),   
            
            nn.MaxPool2d((2, 2), 2)

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



        
if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = LeNet5(1, 32, 16, 10)
    teacher = count_parameters(model)
    model.compress(2)
    student = count_parameters(model)
    print(teacher, student)
    print(teacher/student)
