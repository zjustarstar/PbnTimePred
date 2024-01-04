import torch.nn
import timm
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


class OutputBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(OutputBlock, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 0),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_ch, mid_ch//2, 3, 1, 0),
            nn.BatchNorm2d(mid_ch//2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_ch//2, mid_ch//2, 3, 1, 0),
            nn.BatchNorm2d(mid_ch//2),
            nn.GELU(),
            nn.MaxPool2d(2, 2)
            # nn.Sigmoid() 如果使用sigmoid，值被控制在0-1区间，最后计算损失的结果需要乘5，以变为0-5范围以内的数值
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, mid_ch),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid_ch, out_ch)
        )

    def forward(self, x):
        x = self.out(x)
        b = x.size(0)
        x = x.view(b, -1)
        x = self.fc(x)
        return x


class TimeModel(nn.Module):
    def __init__(self, model_type="mlp"):
        super(TimeModel, self).__init__()
        if model_type == "resnet18":  # 使用resnet前五层网络
            self.model = models.resnet18(pretrained=True)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
            self.base_model = nn.Sequential(*list(self.model.children()))[: 5]

            self.output_model = OutputBlock(64, 32, 1)
        elif model_type == "resnet50":
            self.model = models.resnet101(pretrained=False)
            self.base_model = nn.Sequential(*list(self.model.children()))[: -2]  # 使用resnet除了倒数两层以外的网络
            self.output_model = OutputBlock(2048, 512, 1)
        elif model_type == "mlp":
            mlp1 = nn.Sequential(nn.Linear(17+100+140*2, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1))
            # mlp2 = nn.Sequential(nn.Linear(320, 256), nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(0.1))
            # mlp3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128),nn.ReLU(), nn.Dropout(0.2))
            mlp4 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64),nn.GELU())
            mlp5 = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32),nn.GELU())
            mlp6 = nn.Sequential(nn.Linear(32, 8), nn.BatchNorm1d(8),nn.GELU(), nn.Linear(8, 1))
            self.model = nn.Sequential(mlp1, mlp4, mlp5, mlp6)
            # 使用高斯分布初始化参数
            # self._initialize_weights()
        elif model_type == "mlp_1":
            self.model_input_area = nn.Sequential(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
            self.model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

        elif model_type =="resnet18-mlp":
            self.model = models.resnet18(pretrained=True)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base_model = nn.Sequential(*list(self.model.children()))[: 5]
            self.output_model = OutputBlock(64, 32, 1)

            # self.mlp_model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 16),
            #                            nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 8), nn.ReLU(), nn.Dropout(0.5),
            #                            nn.Linear(8, 1))  # 六层MLP"

            self.mlp_model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16,1))
        elif model_type == "vit":
            self.model = timm.create_model('vit_base_patch32_384',checkpoint_path='./checkpoints/vit/pytorch_model.bin', pretrained=True)
            self.model.head = nn.Linear(self.model.head.in_features, 1)
            self.model.blocks = torch.nn.Sequential( *( list(self.model.blocks)[0:3] ) )#只取前三个block
            self.model.blocks[0].mlp.drop = torch.nn.Dropout(p=0.3, inplace=False)
        else:
            raise Exception("Please select a model")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


    def forward(self, x, model_type="mlp"):
        if model_type == "resnet18":
            x = self.base_model(x)
            # x = self.avg_pool(x)
            x = self.output_model(x)
            return x
        elif model_type == "resnet50":
            x = self.base_model(x)
            # batch, ch = x.size(0), x.size(1)
            # x = self.avg_pool(x).view(batch, ch)
            # x = self.avg_pool(x)
            x = self.output_model(x)
            batch, ch = x.size(0), x.size(1)
            x = x.view(batch, ch)
            return x
        elif model_type == "mlp":
            color_num = x[0]
            blocks_num = x[1]
            blk_per_color = x[2].squeeze(-1)
            area_per_color = x[3].squeeze(-1)
            small_area_num = x[4].squeeze(-1)
            block_distribute = x[5].squeeze(-1)
            # hint= x[5]
            x = torch.cat([color_num, blocks_num, blk_per_color, area_per_color, small_area_num, block_distribute], -1)
            x = self.model(x)
            return x
        elif model_type == "mlp_1":
            # 单独处理面积特征，然后再堆叠
            sehao_feature = x[0]
            sekuai_feature = x[1]
            area_feature = x[2].squeeze(-1)
            area_feature = self.model_input_area(area_feature)
            x = torch.cat([sehao_feature, sekuai_feature, area_feature], -1)
            x = self.model(x)
            return x
        elif model_type =="resnet18-mlp":
            img = x[0]
            sehao_feature = x[1]
            sekuai_feature = x[2]
            img_feature = self.base_model(img)
            img_feature = self.output_model(img_feature)
            x = torch.cat([img_feature, sehao_feature, sekuai_feature], -1)
            # print(x.shape)
            x = self.mlp_model(x)
            return x
        elif model_type == 'vit':
            x = self.model(x)
        elif model_type == "model":
            x = self.base_model(x)
            x = self.max_pool(x)
            x = self.output_model(x)
            batch, ch = x.size(0), x.size(1)
            x = x.view(batch, ch)
            return x
        else:
            raise Exception("Please select a model")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.01)  # 使用高斯分布初始化权重
                init.constant_(m.bias, 0.001)  # 使用常数来初始化偏置

