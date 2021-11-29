from mzutils import parent_dir_and_name, get_things_in_loc, normalize_spaces, denormalize_spaces, mkdir_p
from torch.utils.data import Dataset

class TorchDatasetFromD4RL(Dataset):
    def __init__(self, dataset_d4rl) -> None:
        import d3rlpy
        """
        dataset_d4rl should be a dictionary in the d4rl dataset format.
        """
        self.dataset = d3rlpy.dataset.MDPDataset(dataset_d4rl['observations'], dataset_d4rl['actions'], dataset_d4rl['rewards'], dataset_d4rl['terminals'])
        
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        episode =self.dataset.__getitem__(idx)
        return {'observations': episode.observations, 'actions': episode.actions, 'rewards': episode.rewards}