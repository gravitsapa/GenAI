from datasets import load_dataset
from torch.utils.data import Dataset
from gen_ai.config import DATA_DIR

ImageDataset = Dataset

class AnimeFaces256(ImageDataset):
    def __init__(self, data_dir=DATA_DIR):
        super().__init__()

        self.dataset_dir = DATA_DIR / "anime_faces_256"

        self.data = load_dataset(
            "puruchinera/anime-faces-256",
            cache_dir=self.dataset_dir,
        )

    def __len__(self):
        return len(self.data['train'])

    def __getitem__(self, index):
        return self.data['train'][index]

