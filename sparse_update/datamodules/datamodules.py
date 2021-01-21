import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from .register import register
from datasets import load_dataset


class GlueDataModule(LightningDataModule):
    """
    PyTorch Lightning module for Glue dataset. Need to implement
    `CustomDataset` in child classes.
    """

    name: str

    def __init__(self, batch_size, num_workers, tokenizer):
        """
        Args:
            batch_size: size of each batch
            num_workers: how many workers used to extract data
            tokenizer: Hugginface tokenizer used to tokenizer the texts into
                       tensors.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer

    def prepare_data(self):
        load_dataset("glue", self.name)

    def setup(self, stage=None):
        dataset = load_dataset("glue", self.name)

        self.train_dset, self.val_dset, self.test_dset = (
            dataset["train"],
            dataset["validation"],
            dataset["test"],
        )

        # Create the datasets
        self.train_dset = self.CustomDataset(self.train_dset, self.tokenizer)
        self.val_dset = self.CustomDataset(self.val_dset, self.tokenizer)
        self.test_dset = self.CustomDataset(self.test_dset, self.tokenizer)

    def train_dataloader(self):
        # Only train data will be shuffled
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    class CustomDataset(Dataset):
        def __init__(self, dset, tokenizer):
            super().__init__()

        def __getitem__(self, idx):
            raise NotImplementedError

        def __len__(self):
            raise NotImplementedError


@register
class SST2(GlueDataModule):
    """
    Data module for sst2 dataset. Must initiate `name`
    before __init__() for registering the class.
    """

    name = "sst2"

    def __init__(self, batch_size, num_workers, tokenizer):
        super(SST2, self).__init__(batch_size, num_workers, tokenizer)

    class CustomDataset(Dataset):
        def __init__(self, dset, tokenizer):
            super().__init__()
            self.dset = dset
            self.tokenizer = tokenizer

        def __getitem__(self, idx):

            data = self.dset[idx]["sentence"]
            label = self.dset[idx]["label"]

            # Pad the tensor to max length to make data loaders be able to
            # concatenate aggregated data
            data_dict = self.tokenizer(
                data,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            label = torch.LongTensor([label])

            return (
                data_dict["input_ids"].squeeze(0),
                data_dict["attention_mask"].squeeze(0),
                data_dict["token_type_ids"].squeeze(0),
                label.squeeze(0),
            )

        def __len__(self):
            return len(self.dset)


@register
class QNLI(GlueDataModule):
    """
    Data module for qnli dataset. Must initiate `name`
    before __init__() for registering the class.
    """

    name = "qnli"

    def __init__(self, batch_size, num_workers, tokenizer):
        super(QNLI, self).__init__(batch_size, num_workers, tokenizer)
