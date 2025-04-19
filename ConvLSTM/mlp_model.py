import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self,n=20, m = 4):
        super(mlp,self).__init__()
        layer = 4
        hl = [n,200,150,m]
        
        linear = []
        modules = []
        #nn layers
        for i in range(0,layer-1):
 
            input_size=hl[i]
            output_size=hl[i+1]
            linear = torch.nn.Linear(input_size, output_size)
            bn = torch.nn.BatchNorm1d(output_size)
            modules.append(linear)
            modules.append(bn)
            if i ==layer-2 :
                actFunc = nn.Sigmoid()
                #actFunc = nn.Softmax(dim=1)
            else :
                actFunc = nn.ReLU()
            modules.append(actFunc)
            if i in [1]:
                dropout = nn.Dropout(0.3)
                modules.append(dropout)
            
        self.model = nn.Sequential(*modules)
        
    def init_w_b(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # weight와 bias 초기화
                nn.init.normal_(m.weight, mean=0, std=0.01)
                
                nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        out = self.model(x)
        return out