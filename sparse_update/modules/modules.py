import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification
from pytorch_lightning import LightningModule
from .register import register
from datasets import load_metric
from sparse_update.utilities.optimization import get_scheduler


@register
class SST2Module(LightningModule):
    name = "sst2"

    def __init__(self, args, config_name, tokenizer):
        super().__init__()
        self.args = args
        self.model = BertForSequenceClassification.from_pretrained(config_name)
        self.tokenizer = tokenizer

    def on_epoch_start(self):
        self.metric = load_metric("glue", self.name)

    def on_train_epoch_start(self):
        self.train_metric = load_metric("glue", self.name)

    def on_validation_epoch_start(self):
        self.val_metric = load_metric("glue", self.name)

    def on_test_epoch_start(self):
        self.test_metric = load_metric("glue", self.name)

    def shared_step(self, batch, batch_idx, metric, mode="train"):
        input_ids, attention_mask, token_type_ids, labels = batch

        # truncate the data to maximum length within this batch

        max_len = attention_mask.sum(-1).max()

        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        token_type_ids = token_type_ids[:, :max_len]

        return_dict = self.model(
            input_ids, attention_mask, token_type_ids, return_dict=True, labels=labels
        )

        self.log(f"{mode}/loss", return_dict["loss"])

        predictions = torch.argmax(return_dict["logits"], -1)

        metric.add_batch(predictions=predictions, references=labels)

        return {"loss": return_dict["loss"]}

    def training_step(self, batch, batch_idx):
        self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0])

        return self.shared_step(batch, batch_idx, self.train_metric, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, self.val_metric, "val")

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        # truncate the data to maximum length within this batch

        max_len = attention_mask.sum(-1).max()

        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        token_type_ids = token_type_ids[:, :max_len]

        return_dict = self.model(
            input_ids, attention_mask, token_type_ids, return_dict=True
        )

        predictions = torch.argmax(return_dict["logits"], -1)

        return {"predictions": predictions}

    def shared_epoch_end(self, outputs, metric, mode="train"):
        # acc = torch.cat([o["acc"] for o in outputs], 0)
        # acc = acc.float().mean()

        acc = metric.compute()["accuracy"]

        self.log(f"{mode}/acc", acc)

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, self.train_metric, "train")

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, self.val_metric, "val")

    def test_epoch_end(self, outputs):
        predictions = torch.cat([o["predictions"] for o in outputs], 0)

        out = "index\tprediction\n"

        predictions = predictions.cpu().numpy().tolist()
        indices = list(range(len(predictions)))

        for i, p in zip(indices, predictions):
            out += f"{i}\t{p}\n"

        with open(f"{self.name}.tsv", "w") as record_file:
            record_file.write(out)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.wd,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # num_training_steps = len(self.train_dataloader()) * self.args.max_epochs

        num_training_steps = self.args.max_epochs

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        scheduler_func = get_scheduler(self.args.lr_scheduler_type)
        scheduler = scheduler_func(
            optimizer, self.args.num_warmup_steps, num_training_steps, last_epoch=-1
        )

        return [optimizer], [scheduler]