import torch
from al4pde.acquisition.batch_selection import BatchSelection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Random(BatchSelection):

    def __init__(self, task, data_schedule, batch_size, unc_eval_mode, unc_num_rollout_steps_rel):
        super().__init__(task, data_schedule, batch_size, unc_eval_mode, unc_num_rollout_steps_rel)

    def get_next_params(self, prob_model):
        ic_params = self.task.get_ic_params(self.batch_size)
        pde_params = self.task.get_pde_params_normed(self.batch_size)
        return ic_params, pde_params

    @property
    def name(self):
        return "random"
