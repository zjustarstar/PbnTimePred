import torch.nn
import torch.nn as nn
import torchvision.models as models


class OutputBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(OutputBlock, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_ch, mid_ch//2, 3, 1, 1),
            nn.BatchNorm2d(mid_ch//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_ch//2, mid_ch//2, 3, 1, 1),
            nn.BatchNorm2d(mid_ch//2),
            nn.ReLU(inplace=True)
            # nn.Sigmoid() 如果使用sigmoid，值被控制在0-1区间，最后计算损失的结果需要乘5，以变为0-5范围以内的数值
        )
        self.fc = nn.Sequential(
            nn.Linear(4096*4, mid_ch),
            nn.ReLU(inplace=True),
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

        # if model_type == "resnet18":  # 使用resnet除了倒数两层以外的网络
        #     self.model = models.resnet18(pretrained=False)
        #     self.base_model = nn.Sequential(*list(self.model.children()))[: -2]
        #     self.output_model = OutputBlock(512, 128, 1)
        if model_type == "resnet18":  # 使用resnet前五层网络
            self.model = models.resnet18(pretrained=True)
            # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base_model = nn.Sequential(*list(self.model.children()))[: 5]

            self.output_model = OutputBlock(64, 32, 1)
        elif model_type == "resnet50":
            self.model = models.resnet101(pretrained=False)
            self.base_model = nn.Sequential(*list(self.model.children()))[: -2]  # 使用resnet除了倒数两层以外的网络
            self.output_model = OutputBlock(2048, 512, 1)
        elif model_type == "mlp":
            # self.model = nn.Sequential(nn.Linear(2, 1)) # 一层MLP
            # self.model = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))  # 三层MLP
            # self.model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))  # 六层MLP
            # self.model = nn.Sequential(nn.Linear(23, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))  # 六层MLP
            self.model = nn.Sequential(nn.Linear(243, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 8), nn.ReLU(), nn.Linear(8, 1))
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
            #                            nn.Linear(8, 1))  # 六层MLP

            self.mlp_model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16,1))
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
            hint= x[4]
            x = torch.cat([color_num, blocks_num, blk_per_color, area_per_color, hint], -1)
            print(x.shape)
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
        elif model_type == "model":
            x = self.base_model(x)
            x = self.max_pool(x)
            x = self.output_model(x)
            batch, ch = x.size(0), x.size(1)
            x = x.view(batch, ch)
            return x
        else:
            raise Exception("Please select a model")


