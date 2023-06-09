import os
import random
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image


class StylizedNet(nn.Module):

    def __init__(self, name='resnet50', device='cpu', img_dir=None):
        super(StylizedNet, self).__init__()
        self.name = name
        self.device = device
        self.model = self.load_vanilla_surrogate_model()
        self.vanilla_model = self.load_vanilla_surrogate_model()

        self.styless_num = 10
        self.scale_bound = 1.0
        self.img_dir = img_dir
        self.style_img_list = []
        self.mix_rate = 0.2  # if len(self.style_img_list) > 1 else 0
        self.styless_layer = self.gen_instance_norm_layer()
        self.insert_styless_layer()
        self.std_mean_style_vanilla = None
        self.content_img = None
        self.candidate_layer_num = 200
        self.candidate_layer_list = []
        self.check_acc_flag = True  # maintain original top-1 acc
        self.save_para_flag = False
        self.save_para_dir = None
        self.load_saved_para = False

        self.trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        # self.load_style_img()

    def load_vanilla_surrogate_model(self):
        model = getattr(torchvision.models, self.name)(pretrained=True)
        model.to(self.device)
        model.eval()
        return model

    def gen_instance_norm_layer(self):
        if self.name == 'resnet50' or self.name == 'wide_resnet101_2':
            idx_layer_ = self.get_styless_layer_idx()
            styless_layer = nn.InstanceNorm2d(64 * (2 ** (idx_layer_ - 1)) * 4,
                                              affine=True)
            # init the layer para
            styless_layer.weight.data.fill_(1)
            styless_layer.bias.data.fill_(0)
            styless_layer.to(self.device)
            return styless_layer
        elif self.name == 'densenet121':
            _ = self.get_styless_layer_idx()
            styless_layer = nn.InstanceNorm2d(64, affine=True)
            styless_layer.weight.data.fill_(1)
            styless_layer.bias.data.fill_(0)
            styless_layer.to(self.device)
            return styless_layer
        else:
            raise NotImplementedError

    def get_styless_layer_idx(self):
        if self.name == 'resnet50' or self.name == 'wide_resnet101_2':
            # [3, 4, 6, 3]
            self.idx_layer_ = 1
            self.idx_block_ = 1
            return self.idx_layer_

        elif self.name == 'densenet121':
            # [6,12,24,16]
            self.idx_layer_ = 5
            self.idx_block_ = 1
            return self.idx_layer_
        else:
            raise NotImplementedError

    def init_styless_layer(self, x):
        self.content_img = x

        def get_mid_output(m, i, o):
            global mid_output
            mid_output = o

        if self.name == 'resnet50' or self.name == 'wide_resnet101_2':
            style_input = self.model._modules.get(
                "layer" + str(self.idx_layer_))[self.idx_block_ - 1]
        elif self.name == 'densenet121':
            style_input = self.model._modules.get(
                "features")[self.idx_block_ - 1]

        h = style_input.register_forward_hook(get_mid_output)
        # set mean and variance as the para of styless IN layer
        out = self.model(x)
        mid_original = torch.zeros(mid_output.size())
        mid_original.copy_(mid_output.detach())
        std_mean = torch.std_mean(mid_original, dim=(2, 3),
                                  keepdim=False, unbiased=False)
        std_x, mean_x = std_mean[0][0], std_mean[1][0]
        std_x, mean_x = std_x.to(self.device), mean_x.to(self.device)

        self.styless_layer.weight.data.copy_(std_x)
        self.styless_layer.bias.data.copy_(mean_x)
        self.std_mean_style_vanilla = (std_x, mean_x)

        h.remove()

    def simulate_multiple_layer_para(self, x=None, y=None):
        if self.load_saved_para:
            try:
                self.load_styless_layer_para(y, self.device)
                return
            except:
                raise Exception("Failed to load IN: %s" % "{:05d}.npy".format(y.cpu()[0].item()))

        self.style_img_list = self.load_style_img()
        x = self.content_img if x is None else x
        if self.vanilla_model is None:
            self.vanilla_model = self.load_vanilla_surrogate_model()

        def get_mid_output(m, i, o):
            global mid_output
            mid_output = o

        if self.name == 'resnet50' or self.name == 'wide_resnet101_2':
            style_input = self.model._modules.get(
                "layer" + str(self.idx_layer_))[self.idx_block_ - 1]
        elif self.name == 'densenet121':
            style_input = self.model._modules.get(
                "features")[self.idx_block_ - 1]  # [self.idx_layer_]
        h = style_input.register_forward_hook(get_mid_output)

        # pred_ori = self.model(x, vanilla=True).argmax(dim=-1)
        pred_ori = self.vanilla_model(x).argmax(dim=-1)

        loop_count, loop_max = 0, 1000
        while len(self.candidate_layer_list) != self.candidate_layer_num:
            loop_count += 1
            if loop_count > loop_max:
                # print('Warning: exceed loop_max, simulate %s styless layers' %
                #       len(self.candidate_layer_list))
                break
            std_x, mean_x = self.std_mean_style_vanilla
            mixRate = self.mix_rate
            device = self.device
            # mix the style
            std_x_, mean_x_ = std_x * (1 - mixRate), mean_x * (1 - mixRate)
            mixNum = 1
            for _ in range(mixNum):
                # get a random element from self.style_img_list
                if len(self.style_img_list) > 0:
                    idx = random.randint(0, len(self.style_img_list) - 1)
                    xs_f, xs = self.style_img_list[idx]
                else:
                    std_x_, mean_x_ = std_x, mean_x
                    break
                xs = xs.to(device)
                # get style features
                out = self.model(xs.unsqueeze(0))
                mid_original = torch.zeros(mid_output.size())
                mid_original.copy_(mid_output.detach())
                std_mean_xs = torch.std_mean(mid_original, dim=(2, 3),
                                             keepdim=False, unbiased=False)
                std_xs, mean_xs = std_mean_xs[0][0], std_mean_xs[1][0]
                std_xs, mean_xs = std_xs.to(device), mean_xs.to(device)
                # mix with style input
                # std_x = std_x * (1 - mixRate) + std_xs * mixRate / mixNum
                # mean_x = mean_x * (1 - mixRate) + mean_xs * mixRate / mixNum
                std_x_ += std_xs * mixRate / mixNum
                mean_x_ += mean_xs * mixRate / mixNum
            std_x, mean_x = std_x_, mean_x_
            # scale the style
            scale_std = torch.randn_like(std_x) + 1
            scale_mean = torch.randn_like(mean_x) + 1
            scale_std = torch.clamp(scale_std, 0, self.scale_bound + 1)
            scale_mean = torch.clamp(scale_mean, 0, self.scale_bound + 1)
            std_x = std_x * scale_std
            mean_x = mean_x * scale_mean
            std_x = std_x.to(device)
            mean_x = mean_x.to(device)
            # set styless layer
            self.styless_layer.weight.data.copy_(std_x)
            self.styless_layer.bias.data.copy_(mean_x)
            pred_sty = self.model(x).detach()
            # whether the pred is correct
            acc_flag = torch.equal(pred_sty.argmax(dim=-1), pred_ori)
            if y is not None:
                acc_flag = acc_flag or torch.equal(pred_sty.argmax(dim=-1), y)

            if not self.check_acc_flag:
                self.candidate_layer_list.append((std_x, mean_x))
            elif acc_flag and self.check_acc_flag:
                self.candidate_layer_list.append((std_x, mean_x))

        h.remove()
        # del self.vanilla_model

        # save the simulated styless layer para to npy, with xs_f as index
        if self.save_para_flag:
            # print('begin save para.')
            candidate_layer_list_np = []
            for i, (std_x, mean_x) in enumerate(self.candidate_layer_list):
                std_x = std_x.detach().cpu().numpy()
                mean_x = mean_x.detach().cpu().numpy()
                candidate_layer_list_np.append((std_x, mean_x))
            save_n = "{:05d}".format(y.cpu()[0].item())  # bs=1
            os.makedirs(self.save_para_dir, exist_ok=True)
            save_para_path = os.path.join(self.save_para_dir, save_n + '.npy')
            np.save(save_para_path, candidate_layer_list_np)
            # print('save simulated styless layer para to %s' % save_para_path)

        random.shuffle(self.candidate_layer_list)
        return

    def insert_styless_layer(self):
        if self.name == 'resnet50' or self.name == 'wide_resnet101_2':
            module = self.model._modules['layer%s' % self.idx_layer_]
            module_list = list(module.children())
            module_list.insert(self.idx_block_, self.styless_layer)
            module = nn.Sequential(*module_list)
            # set the new module to the model
            self.model._modules['layer%s' % self.idx_layer_] = module
        elif self.name == 'densenet121':
            module = self.model._modules['features']  # [4]["denselayer%s"%self.idx_layer_]
            module_list = list(module.children())
            module_list.insert(self.idx_block_, self.styless_layer)
            module = nn.Sequential(*module_list)
            # set the new module to the model
            self.model._modules['features'] = module
        else:
            raise NotImplementedError

    def load_style_img(self):
        # load images from img_dir, *.png, or */*.png, or JPEG
        file_list, style_img_list = [], []
        for root, dirs, files in os.walk(self.img_dir):
            for f in files:
                if f.endswith('.png') or f.endswith('.JPEG'):
                    file_list.append(os.path.join(root, f))
        if len(file_list) == 0: return []
        # random pick self.styless_num elements from file_list
        file_list = random.sample(file_list, min(self.styless_num, len(file_list)))
        for img_path in file_list:
            img = Image.open(img_path).convert('RGB')
            style_img_list.append(self.trans(img))
        self.style_img_list = list(zip(file_list, style_img_list))
        return self.style_img_list

    def load_styless_layer_para(self, y, device):
        save_n = "{:05d}".format(y.cpu()[0].item())  # bs=1
        save_para_path = os.path.join(self.save_para_dir, save_n + '.npy')
        candidate_layer_list_np = np.load(save_para_path)
        for std_x, mean_x in candidate_layer_list_np:
            std_x = torch.tensor(std_x).to(device)
            mean_x = torch.tensor(mean_x).to(device)
            self.candidate_layer_list.append((std_x, mean_x))

        random.shuffle(self.candidate_layer_list)
        # print('load simulated styless layer para from %s' % save_para_path)

    def set_styless_layer(self, vanilla=False):
        self.styless_layer.weight.data.copy_(self.std_mean_style_vanilla[0])
        self.styless_layer.bias.data.copy_(self.std_mean_style_vanilla[1])
        if not vanilla and len(self.candidate_layer_list) > 0:
            std_x, mean_x = random.sample(self.candidate_layer_list, 1)[0]
            self.styless_layer.weight.data.copy_(std_x)
            self.styless_layer.bias.data.copy_(mean_x)

    def _reset(self, x, y=None):
        self.content_img = None
        self.candidate_layer_list = []
        self.init_styless_layer(x)
        self.simulate_multiple_layer_para(x, y)

    def forward(self, x, vanilla=False):
        # assert len(x.shape) == 4 and x.shape[0] == 1
        if self.content_img is None:
            self._reset(x)

        if vanilla:
            self.set_styless_layer(vanilla=True)
            self.model.eval()
            return self.model(x)
            # return self.vanilla_model(x)
        else:
            self.set_styless_layer()
            self.model.eval()
            return self.model(x)
