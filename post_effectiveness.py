import os

import numpy as np
from pathlib import Path

base_dir = '/home/sxc180080/data/Project/SlothBomb/results/1221/curve'
save_dir = Path('/home/sxc180080/data/Project/SlothBomb/final_res/effectiveness')
save_dir.mkdir(parents=True, exist_ok=True)


threshold_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
index_list = [0, 2, 4, 6, 8, 10, 12]  # the index of threshold 0.5


for data_name in ['cifar10']:
    for index in index_list:
        data_res = []
        for dynamic in ['separate', 'shallowdeep']:
            one_res = []
            for approach_id in range(6):
                final_res = []
                cnt = 1
                st, ed = 8 * approach_id, (8 * approach_id + 8)
                for backbone in ['vgg16', 'mobilenet', 'resnet56']:
                    for p_rate in [0.05, 0.1, 0.15]:
                        file_name = '{}_{}_{}_{}.csv'.format(dynamic, p_rate, data_name, backbone)
                        file_name = os.path.join(base_dir, file_name)
                        tmp = np.loadtxt(file_name, delimiter=',')

                        final_res.append(float(tmp[index, st:ed][-2:-1]))
                        cnt += 1
                final_res = np.array(final_res).reshape([-1, 1])
                one_res.append(final_res)
            one_res = np.concatenate(one_res, axis=1)
            data_res.append(one_res)
        data_res = np.concatenate(data_res)
        save_path = os.path.join(save_dir, data_name + '_' + str(threshold_list[index]) + '.csv')
        np.savetxt(save_path, data_res, delimiter=',')
