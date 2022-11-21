import numpy as np

from src.Defense import StripDefense
from utils import *


def evaluate_defense(defense_type, data_module, back_model, backdoor, tri_type):
    if defense_type == 0:
        defense = StripDefense(data_module, back_model, backdoor, tri_type, add_trigger)
    else:
        raise NotImplemented
    res = defense.run()
    return res


def main(exp_id):
    seed = get_random_seed()
    dynamic_list = ['separate', 'shallowdeep']
    backbone, data_name = EXP_LIST[exp_id]
    device = torch.device(0)
    base_model_path = Path(get_base_path()).joinpath('model_weights/{}'.format(seed))
    base_model_path.mkdir(parents=True, exist_ok=True)

    save_dir = Path(get_base_path()).joinpath('results/{}/defense/'.format(seed))
    save_dir.mkdir(parents=True, exist_ok=True)

    data_module = get_dataset(data_name)
    for dynamic in dynamic_list:
        for poisoning_rate in [0.05, 0.1, 0.15]:
            backdoor_model_list = load_backdoor_model_list(base_model_path, poisoning_rate, dynamic, exp_id)
            for defense_type in [0]:
                final_res = []
                save_path = save_dir.joinpath('defense-{}_{}_{}_{}_{}.csv'.format(defense_type, dynamic, poisoning_rate, data_name, backbone))

                for i, (back_model, backdoor, tri_type) in enumerate(backdoor_model_list):
                    if i <= 2:
                        continue
                    back_model = back_model.to(device).eval()
                    backdoor = [d.to(device) for d in backdoor]

                    res = evaluate_defense(defense_type, data_module, back_model, backdoor, tri_type)
                    final_res.append(res)
                final_res = np.concatenate(final_res)
                np.savetxt(save_path, final_res, delimiter=',')


            # final_res = np.concatenate(final_res, axis=1)
            # np.savetxt(save_path, final_res, delimiter=',')

if __name__ == '__main__':
    for exp_id in range(3):
        main(exp_id)