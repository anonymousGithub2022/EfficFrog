
from utils import *
from exp_evaluate_backdoor_model import evaluate_backdoor_model

def main(exp_id):
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    backbone, data_name = EXP_LIST[exp_id]
    device = torch.device(0)
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))
    base_model_path.mkdir(parents=True, exist_ok=True)

    save_dir = Path(get_base_path()).joinpath('results/{}/fewshot/'.format(seed))
    save_dir.mkdir(parents=True, exist_ok=True)
    final_res = []
    for dynamic in dynamic_list:
        clean_model = load_clean_model(base_model_path, dynamic, data_name, backbone)
        clean_model = clean_model.to(device).eval()
        poisoning_rate = 0.002

        (attack_name, tri_prefix, tri_type) = ('replace_fix_backdoor.pt', '.trigger', 1)

        model_dir = Path.joinpath(
            base_model_path, 'attack/{}_{}_{}_{}'.format(poisoning_rate, dynamic, data_name, backbone))
        model_path = model_dir.joinpath(attack_name)
        backdoor_model = torch.load(model_path).to('cpu').eval()
        mask, _ = get_backdoor(data_name, trigger_type=0)
        trigger_path = os.path.join(str(TRIGGERDIR), str(tri_type) + '_' + dynamic + '_' + str(exp_id) + tri_prefix)
        trigger = torch.load(trigger_path)

        backdoor_model = backdoor_model.to(device).eval()
        backdoor = [mask, trigger]
        backdoor = [d.to(device) for d in backdoor]

        save_data = evaluate_backdoor_model(clean_model, backdoor_model, backdoor, tri_type, data_name, device)
        final_res.append([save_data[6][0], save_data[6][6]])
    print(final_res)
    # save_path = save_dir.joinpath(str(exp_id) + '.csv')
    # final_res = np.concatenate(final_res, axis=1)
    # np.savetxt(str(save_path), final_res, delimiter=',')


if __name__ == '__main__':
    for i in range(3):
        main(i)
