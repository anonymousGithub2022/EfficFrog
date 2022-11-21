import numpy as np

from utils import *

error_res_dict = {}
blocks_dict = {}
save_dir = Path(get_base_path()).joinpath('intermediate')

for dynamic in DYNAMICISM:
    for exp_subj in EXP_LIST[:3]:
        backbone, dataset = exp_subj

        subj_name = dynamic + '_' + backbone + '_' + dataset
        save_name = str(save_dir.joinpath(subj_name))
        res = torch.load(save_name)
        acc, avg_blocks, pred_list, block_list, y_list = res
        error_res_dict[subj_name] = np.where(res[2] != res[-1])[0]
        blocks_dict[subj_name] = res[3].numpy()

acc_matrix = np.zeros([len(error_res_dict), len(error_res_dict)])
block_matrix = np.zeros([len(error_res_dict), len(error_res_dict)])

for i, m in enumerate(error_res_dict):
    for j, n in enumerate(error_res_dict):
        common = set(error_res_dict[m].tolist()).intersection(set(error_res_dict[n].tolist()))
        supper = set(error_res_dict[m].tolist()).union(set(error_res_dict[n].tolist()))
        acc_matrix[i][j] = len(common) / len(supper)

        block_matrix[i][j] = np.corrcoef(blocks_dict[m], blocks_dict[n])[1][0]

print(acc_matrix)

print(block_matrix)