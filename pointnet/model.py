import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))
        
    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        # Input transform
        x = pointcloud.clone().detach() #[B,N,3]
        if self.input_transform:
            x = torch.transpose(x, 1, 2) #[B,3,N]
            transform1 = self.stn3(x) #[3,3]
            x = torch.transpose(x, 1, 2) #[B,N,3]
            x = torch.bmm(x, transform1) #[B,N,3]
        else:
            transform1 = None
            
        # mlp(64,64)
        x = torch.transpose(x, 1, 2) #[B,3,N]
        x = F.relu(self.conv1(x)) #[B,64,N]

        # Feature transform
        if self.feature_transform:
            transform2 = self.stn64(x) #[64,64]
            x = torch.transpose(x, 1, 2) #[B,N,64]
            x = torch.bmm(x, transform2) #[B,N,64]
            local_feature = x.clone().detach() # For Part Segmentation
            x = torch.transpose(x, 1, 2) #[B,64,N]
        else:
            transform2 = None
        
        # mlp(64,128,1024)
        x = F.relu(self.conv2(x)) #[B,128,N]
        x = self.conv3(x) #[B,1024,N]

        # maxpool
        x = torch.max(x, 2)[0] #[B,1024]
        global_feature = x
        
        return global_feature, local_feature, transform1, transform2


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        global_feature, _, transform1, transform2 = self.pointnet_feat(pointcloud) #[B,1024]
        output_scores = self.mlp(global_feature) #[B, num_classes]
        output_scores = F.log_softmax(output_scores, dim=-1) #[B, num_classes]
        return output_scores, transform1, transform2


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.m = m
        self.pointnet_feat = PointNetFeat(True, True)
        
        self.conv1 = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512))
        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128))
        self.conv4 = nn.Sequential(nn.Conv1d(128, self.m, 1), nn.BatchNorm1d(self.m))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        global_feature, local_feature, transform1, transform2 = self.pointnet_feat(pointcloud) #[B,1024], [B, N, 64]
        
        _, N, _ = local_feature.shape
        B, k = global_feature.shape #k=1024
        global_feature = global_feature.expand(N, B, k).transpose(1, 0) #[B, N, 1024]
        x = torch.cat((local_feature, global_feature), dim=-1) #[B, N, 1088]
        
        # mlp(512,256,128)
        x = torch.transpose(x, 1, 2) #[B,1088,N]
        x = F.relu(self.conv1(x)) #[B,512,N]
        x = F.relu(self.conv2(x)) #[B,256,N]
        x = F.relu(self.conv3(x)) #[B,128,N]
        
        # mlp(128, m)
        x = self.conv4(x) #[B,m,N]
        
        # logit
        x = torch.transpose(x, 1, 2) #[B,N,m]
        x = F.log_softmax(x, dim=-1)
        return x, transform1, transform2


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.num_points = num_points
        
        self.mlp = nn.Sequential(
            nn.Linear(1024, self.num_points//4),
            nn.BatchNorm1d(self.num_points//4),
            nn.ReLU(),
            nn.Linear(self.num_points//4, self.num_points//2),
            nn.BatchNorm1d(self.num_points//2),
            nn.ReLU(),
            nn.Linear(self.num_points//2, self.num_points),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(self.num_points),
            nn.ReLU(),
            nn.Linear(self.num_points, self.num_points*3),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        B, N, _ = pointcloud.shape
        global_feature, _, _, _ = self.pointnet_feat(pointcloud)
        
        output = self.mlp(global_feature)
        return output.reshape(B, N, 3)        


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
