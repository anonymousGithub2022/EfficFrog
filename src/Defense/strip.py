import cv2
import numpy as np
import torch
from tqdm import tqdm

class StripDefense:
    def __init__(self, data_module, model, backdoor, tri_type, add_trigger_func):
        self.data_module = data_module
        self.model = model.eval().to('cuda')
        self.backdoor = backdoor
        self.tri_type = tri_type
        self.add_trigger_func = add_trigger_func
        self.normalize = data_module.trigger_normalized

        self.train_loader = self.data_module.train_loader
        self.test_loader = self.data_module.test_loader

        self.test_x, self.test_y = [], []
        for (x, y) in self.test_loader:
            self.test_x.append(x)
            self.test_y.append(y)
        self.test_x = torch.cat(self.test_x)
        self.test_y = torch.cat(self.test_y)

    def superimpose(self, background, overlay):
        overlay = overlay.to(background.device)
        added_image = background + overlay
        return added_image

    def entropyCal(self, background, n):
        x1_add = []
        index_overlay = np.random.randint(0, 10000, size=n)
        for x in range(n):
            tmp = self.test_x[index_overlay[x]].unsqueeze(0)
            x1_add.append(self.superimpose(background, tmp))
        x1_add = torch.cat(x1_add)
        py1_add = self.model(x1_add)[-1]
        score = py1_add * torch.log2(py1_add)
        score = score.detach().cpu().numpy()
        EntropySum = -np.nansum(score)
        return EntropySum

    def run(self):
        n_test = 1000
        n_sample = 100
        entropy_benign = [0] * n_test
        entropy_trojan = [0] * n_test

        for j, x_background in tqdm(enumerate(self.test_x)):
            if j == n_test:
                break
            x_background = x_background.unsqueeze(0)
            x_background = x_background.to('cuda')
            x_poison = self.add_trigger_func(x_background, self.backdoor, poisoning_rate=1, normalize=self.normalize, trigger_type= self.tri_type)

            entropy_benign[j] = self.entropyCal(x_background, n_sample)
            entropy_trojan[j] = self.entropyCal(x_poison, n_sample)

        entropy_benign = [x / n_sample for x in entropy_benign]  # get entropy for 2000 clean inputs
        entropy_trojan = [x / n_sample for x in entropy_trojan]  # get entropy for 2000 trojaned inputs
        res = np.concatenate([np.array(entropy_benign).reshape([-1, 1]), np.array(entropy_trojan).reshape([-1, 1])], axis=1)
        return res
