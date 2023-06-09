import os
import copy
import tqdm
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import scipy.stats as st
from functools import partial
from utils import renormalization, UnNormalize, im_dataset
from stylized_model import StylizedNet

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)])


def styless_attack(model_pool,
                   X,
                   y,
                   niters=50,
                   epsilon=16 / 255.,
                   learning_rate=8 / 255.,
                   mi=False,
                   mi_decay=1.0,
                   ti=False,
                   di=False,
                   si=False,
                   admix=False,
                   admix_dir=None,
                   device="cpu"):
    for model in model_pool:
        model.eval()

    if ti: ti_conv = gen_ti_conv(device)
    if mi: momentum = 0
    if admix: admix_img_list = get_admix_img(admix_dir, device)
    scale_list = 1. / torch.tensor([1., 2., 4., 8., 16.]) if si else torch.tensor([1.])
    scale_list = scale_list.to(device)
    admix_size = 1 if not admix else 3

    X_pert = X.clone()
    X_pert.requires_grad = True
    for i in range(niters):
        y_used = y
        ll_factor = 1  # non-targeted attack

        gradient = 0
        # X_pert_input = []
        for _ in range(admix_size):
            idx = random.randint(0, len(admix_img_list) - 1) if admix else None
            X_admix_add = torch.tensor(0).to(device) if not admix else 0.2 * admix_img_list[idx]
            X_admix_add_tensor = X_admix_add.repeat(scale_list.size(0), 1, 1, 1)
            X_pert_batch = X_pert.detach().clone().repeat(scale_list.size(0), 1, 1, 1)
            X_pert_batch.requires_grad = True
            X_pert_input = (X_pert_batch + X_admix_add_tensor) * scale_list[:, None, None, None]

            X_pert_input = input_diversity(X_pert_input) if di else X_pert_input
            total_loss = 0
            for model in model_pool:
                # match the batch size of X_pert_input
                y_used_ = y_used.repeat(X_pert_input.shape[0])
                loss = nn.CrossEntropyLoss()(model(X_pert_input), y_used_)
                total_loss += loss

            total_loss.backward()
            gradient += torch.sum(X_pert_batch.grad.detach() * scale_list[:, None, None, None],
                                  dim=0, keepdim=True)
        gradient /= admix_size

        gradient = ti_conv(gradient) if ti else gradient
        if not mi:
            pert = ll_factor * learning_rate * gradient.sign()
        else:
            momentum = mi_decay * momentum + gradient / torch.mean(
                torch.abs(gradient), dim=(1, 2, 3), keepdim=True)
            pert = learning_rate * momentum.sign()

        X_pert = X_pert.detach() + pert
        # make sure the values are within the epsilon and [0,255] restrictions.
        X_pert = renormalization(X, X_pert, epsilon)
        X_pert.requires_grad = True

    return X_pert


def get_admix_img(img_dir, device='cpu', bs=8):
    file_list, img_list = [], []
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            if f.endswith('.png') or f.endswith('.JPEG'):
                file_list.append(os.path.join(root, f))
    file_list = random.sample(file_list, min(bs, len(file_list)))
    for img_path in file_list:
        img = Image.open(img_path).convert('RGB')
        img = trans(img).to(device)
        img_list.append(img)

    return img_list


# di
def input_diversity(X, p=0.5, image_width=224, image_resize=244):
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = nn.functional.interpolate(X, [rnd, rnd])
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_width, image_width])

    return padded if torch.rand(()) < p else X


# ti
def gen_ti_conv(device):
    def gkern(kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    kernel_size = 5
    kernel = gkern(kernel_size, 3).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)
    ti_conv = partial(F.conv2d, weight=gaussian_kernel, bias=None,
                      stride=1, padding=(2, 2), groups=3)

    return ti_conv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform StyLess Attack.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", default="resnet50", help="model name",
                        choices=["resnet50", "wide_resnet101_2", "densenet121"])
    parser.add_argument("--styless", action='store_true', help="perform StyLess attack")
    parser.add_argument("--save_in", action='store_true', help="store IN layer")
    parser.add_argument("--load_in", action='store_true', help="load IN layer")
    parser.add_argument("--styNum", type=int, default=10, help="number of styles")
    parser.add_argument("--mi", action='store_true', help="perform MI-FGSM attack")
    parser.add_argument("--ti", action='store_true', help="perform TI-FGSM attack")
    parser.add_argument("--di", action='store_true', help="perform DI-FGSM attack")
    parser.add_argument("--si", action='store_true', help="perform SI-FGSM attack")
    parser.add_argument("--admix", action='store_true', help="perform Admix attack")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--img_dir", type=str, default="./data/test_samples", help="image directory")
    parser.add_argument("--exp_name", type=str, default="ifgsm", help="prefix for experiment name")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    device = args.device
    img_dir = args.img_dir
    model_name = args.model
    StyLess = args.styless
    mi = args.mi
    ti = args.ti
    si = args.si
    di = args.di
    admix = args.admix
    admix_dir = None if not admix else img_dir
    att_names = ["mi", "ti", "di", "si", "admix", "styless"]
    exp_name = args.exp_name
    for att_name in att_names:
        if getattr(args, att_name):
            exp_name += "_" + att_name
    if args.styNum != 10:
        exp_name += "_" + str(args.styNum)
    exp_dir = f'exp/{os.path.basename(args.img_dir)}/{args.model}/{exp_name}'
    exp_dir_ckpt = f'exp/{os.path.basename(args.img_dir)}/{args.model}/ckpt'
    print("exp_dir: ", exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_dir + "/adv_imgs", exist_ok=True)

    vanilla_net = getattr(torchvision.models, model_name)(pretrained=True).to(device)
    vanilla_net.eval()

    stylized_net = StylizedNet(model_name, device, img_dir)
    stylized_net.eval()

    if args.load_in:
        stylized_net.load_saved_para = True
        stylized_net.save_para_dir = exp_dir_ckpt
    elif args.save_in:
        stylized_net.save_para_flag = True
        stylized_net.save_para_dir = exp_dir_ckpt

    data_test = im_dataset(root=img_dir, transform=trans)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=False)

    model_pool = [vanilla_net]
    unnorm = UnNormalize(mean=mean, std=std)
    for x, y in tqdm.tqdm(test_loader):
        del model_pool[1:]  # only keep the vanilla model
        x, y = x.to(device), y.to(device)
        if StyLess:  # generate stylized models for each image
            stylized_net._reset(x, y)
            for _ in range(args.styNum):
                model_pool.append(copy.deepcopy(stylized_net))

        adv = styless_attack(model_pool, x, y, device=device,
                             mi=mi, ti=ti, si=si, di=di, admix=admix, admix_dir=admix_dir)
        for i in range(len(adv)):
            save_f = os.path.join(exp_dir, "adv_imgs", "{:05d}.png".format(y.cpu()[i].item()))
            torchvision.utils.save_image(unnorm(adv[i]), save_f)

    print("save to: ", exp_dir)
    print('DONE')
