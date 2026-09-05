import torch
from torch.utils.data import Dataset

from pathlib import Path

PAD_ID = 0
IGNORE_ID = -100

DATA_DIR = Path.cwd().parent / "preparation"/ "data"



class FTDataset(Dataset):


    def __init__(self, path: str | Path):
        ds = torch.load(path)

        self.examples: list[dict[str, list[int] | int]] = ds["examples"]
        self.context_length = ds["context_length"]


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):

        example = self.examples[idx]
        ids = example["ids"]
        prompt_len = example["prompt_len"]

        input_ids = ids[:-1]
        target_ids = ids[1:]
        size = len(input_ids)

        target_ids = [IGNORE_ID] * (prompt_len - 1) + target_ids[prompt_len - 1:]

        padding = self.context_length - size
        input_ids = input_ids + padding * [PAD_ID]
        target_ids = target_ids + padding * [IGNORE_ID]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def get_dataset(split: str) -> FTDataset:
    splits = ["train", "val", "test"]
    if split not in splits:
        raise ValueError(f"split must be one of {sorted(splits)}, got {split}")

    return FTDataset(DATA_DIR / f"{split}.pt")