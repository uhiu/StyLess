import torch
import argparse
import torchvision
import pandas as pd
import os
import tqdm
import torchvision.transforms as transforms
from utils import im_dataset


def load_model(model_names=None):
    if model_names is None:
        model_names = ['vgg19', 'resnet18', 'resnet50',
                       'densenet121', 'inception_v3', 'wide_resnet101_2',
                       'mobilenet_v2', 'shufflenet_v2_x1_0',
                       'vit_b_16', 'vit_l_16']

    # load models
    models = []
    for model_name in model_names:
        if model_name == 'vit_l_16':
            wname = 'IMAGENET1K_SWAG_LINEAR_V1'
            model = getattr(torchvision.models, model_name)(weights=wname)
        elif 'inception' in model_name:
            model = torchvision.models.__dict__[model_name](pretrained=True,
                                                            transform_input=False)
        else:
            model = torchvision.models.__dict__[model_name](pretrained=True)
        # model = model.to(args.device)
        model.eval()
        models.append(model)
    print("load models: ", model_names)
    return models, model_names


def get_csv_for_save(data_root, model_names=None, suffix=""):
    save_csv_dir = data_root
    csv_name_ = 'eval_unsecure%s.csv' % suffix
    save_csv_p = os.path.join(save_csv_dir, csv_name_)  # if csv==None else csv
    assert os.path.exists(save_csv_dir), f"{save_csv_dir} not exists"

    if os.path.exists(save_csv_p):
        df = pd.read_csv(save_csv_p, index_col=0)
    else:
        if model_names is None:
            model_names = ['vgg19', 'resnet18', 'resnet50',
                           'densenet121', 'inception_v3', 'wide_resnet101_2',
                           'mobilenet_v2', 'shufflenet_v2_x1_0',
                           'vit_b_16', 'vit_l_16']
        df = pd.DataFrame()
        df = df.reindex(columns=model_names)
        df.to_csv(save_csv_p)

    return df, save_csv_p


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Eval on unsecured models.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default='', help="dir to save csv")
    parser.add_argument("--img_dir", type=str, default='')
    parser.add_argument("--csv_suffix", type=str, default='')
    # parser.add_argument("--csv_path", type=str, default=None)

    args = parser.parse_args()
    args.img_dir = os.path.join(args.save_dir, 'adv_imgs') if args.img_dir == '' else args.img_dir

    model_names = ['vgg19', 'resnet50', 'wide_resnet101_2',
                   'densenet121', 'inception_v3',
                   'mobilenet_v2', 'shufflenet_v2_x1_0']
    models, model_names = load_model(model_names)

    args.attack_dir = os.path.basename(args.img_dir)

    df, save_csv_p = get_csv_for_save(args.save_dir, model_names=model_names,
                                      suffix=args.csv_suffix)
    if not os.path.exists(args.img_dir):
        args.img_dir = os.path.join(args.save_dir, args.img_dir)
    assert os.path.exists(args.save_dir), f"{args.save_dir} not exists"

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean, std)
    ])
    tran_norm = transforms.Normalize(mean, std)

    if len(os.listdir(args.img_dir)) == 0:
        print(f"{args.img_dir} has no *.png, pass")
        exit()
    # data_test = torchvision.datasets.ImageFolder(root=args.img_dir, transform=trans)
    data_test = im_dataset(root=args.img_dir, transform=trans)
    # assert len(data_test) ==1000, f"{args.img_dir} %s"%len(data_test)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=min(50, len(data_test)),
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=False)
    rerun_name = None
    for model_idx in range(len(model_names)):
        model_name = model_names[model_idx]
        save_index = args.attack_dir + "_" + str(len(data_test))
        print(model_name)
        if save_index in df.index:
            if 0 <= df.loc[save_index, model_name] <= 1:
                # break
                print(f"Exist and pass {save_index} - {model_name} acc: ",
                      df.loc[save_index, model_name])
                if model_name == rerun_name:
                    print(f"Rerun eval on {model_name} ...")
                else:
                    continue
        else:
            print(f"Performing evaluation of {args.img_dir} ...")
        model = models[model_idx]
        model.eval()
        model = model.to(args.device)
        acc = 0
        for x, y in tqdm.tqdm(test_loader):
            x = x.to(args.device)
            y = y.to(args.device)
            tr = transforms.Compose([transforms.Resize(299), tran_norm]) if 'inception' in model_name else tran_norm
            out = torch.argmax(model(tr(x)), dim=1)
            acc += torch.sum(out == y).item()
        print(f"{save_index} - {model_name} accuracy: {acc / len(test_loader.dataset)}")
        df.loc[save_index, model_name] = acc / len(test_loader.dataset)
        df.to_csv(save_csv_p)
    print("csv saved to: ", save_csv_p)

    print("DONE")
