import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from .register import register
from datasets import load_dataset


class GLUEDataModule(LightningDataModule):
    """
    PyTorch Lightning module for GLUE dataset. Need to implement
    `CustomDataset` in child classes.
    """

    name: str

    def __init__(self, batch_size, max_seq_length, num_workers, tokenizer):
        """
        Args:
            batch_size: size of each batch
            num_workers: how many workers used to extract data
            tokenizer: Hugginface tokenizer used to tokenizer the texts into
                       tensors.
        """
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
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
        self.train_dset = self.CustomDataset(
            self.train_dset, self.tokenizer, self.max_seq_length
        )
        self.val_dset = self.CustomDataset(
            self.val_dset, self.tokenizer, self.max_seq_length
        )
        self.test_dset = self.CustomDataset(
            self.test_dset, self.tokenizer, self.max_seq_length
        )

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
        def __init__(self, dset, tokenizer, max_seq_length):
            super().__init__()

        def __getitem__(self, idx):
            raise NotImplementedError

        def __len__(self):
            raise NotImplementedError


@register
class SST2(GLUEDataModule):
    """
    Data module for sst2 dataset. Must initiate `name`
    before __init__() for registering the class.
    """

    name = "sst2"

    def __init__(self, batch_size, max_seq_length, num_workers, tokenizer):
        super(SST2, self).__init__(batch_size, max_seq_length, num_workers, tokenizer)

    class CustomDataset(Dataset):
        def __init__(self, dset, tokenizer, max_seq_length):
            super().__init__()
            self.dset = dset
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length

        def __getitem__(self, idx):

            data = self.dset[idx]["sentence"]
            label = self.dset[idx]["label"]

            # Pad the tensor to max length to make data loaders be able to
            # concatenate aggregated data
            data_dict = self.tokenizer(
                data,
                max_length=self.max_seq_length,
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
class QNLI(GLUEDataModule):
    """
    Data module for qnli dataset. Must initiate `name`
    before __init__() for registering the class.
    """

    name = "qnli"

    def __init__(self, batch_size, max_seq_length, num_workers, tokenizer):
        super(QNLI, self).__init__(batch_size, max_seq_length, num_workers, tokenizer)

    class CustomDataset(Dataset):
        def __init__(self, dset, tokenizer, max_seq_length):
            super().__init__()
            self.dset = dset
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length

        def __getitem__(self, idx):

            data_dict = self.tokenizer(
                self.dset[idx]["question"],
                self.dset[idx]["sentence"],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            label = torch.LongTensor([self.dset[idx]["label"]])

            return (
                data_dict["input_ids"].squeeze(0),
                data_dict["attention_mask"].squeeze(0),
                data_dict["token_type_ids"].squeeze(0),
                label.squeeze(0),
            )

        def __len__(self):
            return len(self.dset)


@register
class MNLI(GLUEDataModule):
    """
    Data module for mnli_mismatched dataset. Must initiate `name`
    before __init__() for registering the class.
    """

    name = "mnli"

    def __init__(self, batch_size, max_seq_length, num_workers, tokenizer):
        super(MNLI, self).__init__(batch_size, max_seq_length, num_workers, tokenizer)

    def prepare_data(self):
        load_dataset("glue", self.name)
        load_dataset("glue", "ax")

    def setup(self, stage=None):
        dataset = load_dataset("glue", self.name)
        dataset_ax = load_dataset("glue", "ax")

        self.train_dset = dataset["train"]

        self.val_dset_m, self.test_dset_m = (
            dataset["validation_matched"],
            dataset["test_matched"],
        )

        self.val_dset_mm, self.test_dset_mm = (
            dataset["validation_mismatched"],
            dataset["test_mismatched"],
        )

        self.test_dset_ax = dataset_ax["test"]

        # Create the datasets
        self.train_dset = self.CustomDataset(
            self.train_dset, self.tokenizer, self.max_seq_length
        )
        self.val_dset_m = self.CustomDataset(
            self.val_dset_m, self.tokenizer, self.max_seq_length
        )
        self.val_dset_mm = self.CustomDataset(
            self.val_dset_mm, self.tokenizer, self.max_seq_length
        )
        self.test_dset_m = self.CustomDataset(
            self.test_dset_m, self.tokenizer, self.max_seq_length
        )
        self.test_dset_mm = self.CustomDataset(
            self.test_dset_mm, self.tokenizer, self.max_seq_length
        )
        self.test_dset_ax = self.CustomDataset(
            self.test_dset_ax, self.tokenizer, self.max_seq_length
        )

    def train_dataloader(self):
        # Only train data will be shuffled
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):

        val_dloader_m = DataLoader(
            self.val_dset_m,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        val_dloader_mm = DataLoader(
            self.val_dset_mm,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return [val_dloader_m, val_dloader_mm]

    def test_dataloader(self):
        test_dloader_m = DataLoader(
            self.test_dset_m,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        test_dloader_mm = DataLoader(
            self.test_dset_mm,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        test_dloader_ax = DataLoader(
            self.test_dset_ax,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return [test_dloader_m, test_dloader_mm, test_dloader_ax]

    class CustomDataset(Dataset):
        def __init__(self, dset, tokenizer, max_seq_length):
            super().__init__()
            self.dset = dset
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length

        def __getitem__(self, idx):
            # Pad the tensor to max length to make data loaders be able to
            # concatenate aggregated data
            data_dict = self.tokenizer(
                self.dset[idx]["premise"],
                self.dset[idx]["hypothesis"],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            label = torch.LongTensor([self.dset[idx]["label"]])

            return (
                data_dict["input_ids"].squeeze(0),
                data_dict["attention_mask"].squeeze(0),
                data_dict["token_type_ids"].squeeze(0),
                label.squeeze(0),
            )

        def __len__(self):
            return len(self.dset)


@register
class RTE(GLUEDataModule):
    """
    Data module for rte dataset. Must initiate `name`
    before __init__() for registering the class.
    """

    name = "rte"

    def __init__(self, batch_size, max_seq_length, num_workers, tokenizer):
        super(RTE, self).__init__(batch_size, max_seq_length, num_workers, tokenizer)

    class CustomDataset(Dataset):
        def __init__(self, dset, tokenizer, max_seq_length):
            super().__init__()
            self.dset = dset
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length

        def __getitem__(self, idx):
            # Pad the tensor to max length to make data loaders be able to
            # concatenate aggregated data
            data_dict = self.tokenizer(
                self.dset[idx]["sentence1"],
                self.dset[idx]["sentence2"],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            label = torch.LongTensor([self.dset[idx]["label"]])

            return (
                data_dict["input_ids"].squeeze(0),
                data_dict["attention_mask"].squeeze(0),
                data_dict["token_type_ids"].squeeze(0),
                label.squeeze(0),
            )

        def __len__(self):
            return len(self.dset)


@register
class WNLI(GLUEDataModule):
    """
    Data module for wnli dataset. Must initiate `name`
    before __init__() for registering the class.
    """

    name = "wnli"

    def __init__(self, batch_size, max_seq_length, num_workers, tokenizer):
        super(WNLI, self).__init__(batch_size, max_seq_length, num_workers, tokenizer)


@register
class AX(GLUEDataModule):
    """
    Data module for ax dataset. Must initiate `name`
    before __init__() for registering the class.
    """

    name = "ax"

    def __init__(self, batch_size, max_seq_length, num_workers, tokenizer):
        super(AX, self).__init__(batch_size, max_seq_length, num_workers, tokenizer)

    class CustomDataset(Dataset):
        def __init__(self, dset, tokenizer, max_seq_length):
            super().__init__()
            self.dset = dset
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length

        def __getitem__(self, idx):
            # Pad the tensor to max length to make data loaders be able to
            # concatenate aggregated data

            print(self.dset[idx])
            data_dict = self.tokenizer(
                self.dset[idx]["premise"],
                self.dset[idx]["hypothesis"],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            label = torch.LongTensor([self.dset[idx]["label"]])

            return (
                data_dict["input_ids"].squeeze(0),
                data_dict["attention_mask"].squeeze(0),
                data_dict["token_type_ids"].squeeze(0),
                label.squeeze(0),
            )

        def __len__(self):
            return len(self.dset)
