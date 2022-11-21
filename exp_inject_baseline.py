import argparse
import pickle
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2

from utils import *


def train_badnet_model(dynamic, dataset, backbone, base_model_path, poisoning_rate, device):
    current_model_path = Path.joinpath(base_model_path, 'attack/{}_{}_{}_{}'.format(poisoning_rate, dynamic, dataset, backbone))
    Path(current_model_path).mkdir(parents=True, exist_ok=True)

    clean_model_dir = Path.joinpath(base_model_path, 'clean/{}_{}_{}'.format(dynamic, dataset, backbone))
    clean_model_path = str(clean_model_dir.joinpath('model.pt'))
    clean_model = torch.load(clean_model_path)

    config_path = str(clean_model_dir.joinpath('parameters_last'))

    model_config = torch.load(config_path)

    mask, trigger = get_backdoor(dataset, trigger_type=0)
    mask, trigger = mask.to(device), trigger.to(device)
    backdoor = [mask, trigger]
    fig_path = './img/%s.png' % dynamic

    trained_model, model_config = badnet_attack(fig_path, clean_model, model_config, backdoor, poisoning_rate=poisoning_rate, device=device)

    save_path = current_model_path.joinpath('badnet.pt')
    torch.save(trained_model, save_path)


def train_trojan_model(dynamic, dataset, backbone, base_model_path, poisoning_rate, device):
    current_model_path = Path.joinpath(base_model_path, 'attack/{}_{}_{}_{}'.format(poisoning_rate, dynamic, dataset, backbone))
    Path(current_model_path).mkdir(parents=True, exist_ok=True)

    clean_model_dir = Path.joinpath(base_model_path, 'clean/{}_{}_{}'.format(dynamic, dataset, backbone))
    clean_model_path = str(clean_model_dir.joinpath('model.pt'))
    clean_model = torch.load(clean_model_path)

    config_path = str(clean_model_dir.joinpath('parameters_last'))

    model_config = torch.load(config_path)

    mask, trigger = get_backdoor(dataset, trigger_type=0)
    mask, trigger = mask.to(device), trigger.to(device)
    backdoor = [mask, trigger]
    fig_path = './img/%s.png' % dynamic

    trained_model, model_config = trojan_attack(fig_path, clean_model, model_config, backdoor, poisoning_rate=poisoning_rate, device=device)
    save_path = current_model_path.joinpath('trojan.pt')
    torch.save(trained_model, save_path)


def main(exp_id, baseline):
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    backbone, dataset = EXP_LIST[exp_id]
    device = torch.device('cuda')
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))
    base_model_path.mkdir(parents=True, exist_ok=True)
    for dynamic in dynamic_list:
        for poisoning_rate in [0.05, 0.1, 0.15]:
            if baseline == 0:
                train_badnet_model(dynamic, dataset, backbone, base_model_path, poisoning_rate, device)
            else:
                train_trojan_model(dynamic, dataset, backbone, base_model_path, poisoning_rate, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=1, type=int, help='exp id')
    parser.add_argument('--baseline', default=1, type=int, help='exp id')
    args = parser.parse_args()
    main(args.exp, args.baseline)