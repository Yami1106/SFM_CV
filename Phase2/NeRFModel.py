import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L,use_pe=True):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################

        '''
        Initialize NeRF MLP architecture 

        Input :
        embed_pos_L : number of frequencies for position encoding
        embed_direction_L : number of frequencies for direction encoding

        Output :
        None
        '''
        self.use_pe = use_pe
        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L

        hidden_dimensions = 256
        # original xyz(3) + 3(coordinates) * 2(sin and cos values per freq) *L frequencies
        #self.pos_dimensions = 3 + 3*2*embed_pos_L
        self.pos_dimensions = (3 + 3*2*embed_pos_L) if use_pe else 3
        
        # original direction(3) + 3(coordinates) * 2(sin and cos values per freq) *L frequencies
        #self.dir_dimensions = 3 + 3*2*embed_direction_L
        self.dir_dimensions = (3 + 3*2*embed_direction_L) if use_pe else 3
        # 8-layers of  MLP 

        self.fc1 = nn.Linear(self.pos_dimensions,hidden_dimensions)
        self.fc2 = nn.Linear(hidden_dimensions,hidden_dimensions)
        self.fc3 = nn.Linear(hidden_dimensions,hidden_dimensions)
        self.fc4 = nn.Linear(hidden_dimensions,hidden_dimensions)

        # add original position to layer 5 with the abstract developed till layer 4 
        self.fc5 = nn.Linear(hidden_dimensions + self.pos_dimensions,hidden_dimensions)
        self.fc6 = nn.Linear(hidden_dimensions,hidden_dimensions)
        self.fc7 = nn.Linear(hidden_dimensions,hidden_dimensions)
        self.fc8 = nn.Linear(hidden_dimensions,hidden_dimensions)

        # sigma(volume density)
        self.sigma_out = nn.Linear(hidden_dimensions,1)

        # since color depends on viewing direction along with the position
        # feature output is like a summary which will be passed to color 
        self.feature_output = nn.Linear(hidden_dimensions,hidden_dimensions)

        # direction branch
        self.fc_dir = nn.Linear(hidden_dimensions + self.dir_dimensions,128)

        #rgb output 
        self.rgb_output = nn.Linear(128,3)


    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################

        '''
        Expand the coordinates into a higher dimensional vector with the help of sin and cos fucntions

        Input:
        x: 3D coordinates (position or direction)
        L: number of frequencies for encoding

        Output:
        y: concat of original coordinates and the sin and cos values for each frequency
        '''

        out = [x]
        for i in range(L):
            freq = 2**i
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        y = torch.cat(out, dim=-1)
        return y

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################

        '''
        Run full forward pass of NeRF MLP to get density and color 

        Input:
        pos: 3D coordinates of the point in space
        direction: viewing direction of the ray passing through the point

        Output:
        output: concat of rgb and sigma values for input position and direction
        '''

        # position encoding and direction encoding

        pos_enc = self.position_encoding(pos,self.embed_pos_L)
        dir_enc = self.position_encoding(direction,self.embed_direction_L)
        # if self.use_pe:
        #     pos_enc = self.position_encoding(pos, self.embed_pos_L)
        #     dir_enc = self.position_encoding(direction, self.embed_direction_L)
        # else:
        #     pos_enc = pos
        #     dir_enc = direction

        # position branch 
        x = F.relu(self.fc1(pos_enc))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # skip connection 
        x = torch.cat([x,pos_enc],dim=-1)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))

        # volume density output sigma 

        sigma = F.relu(self.sigma_out(x))

        # feature output for color branch
        feature = F.relu(self.feature_output(x))


        # direction branch
        h = torch.cat([feature,dir_enc],dim=-1)
        h = F.relu(self.fc_dir(h))
        rgb = torch.sigmoid(self.rgb_output(h))

        output = torch.cat([rgb,sigma],dim=-1)
        return output
