import os
import torch
import argparse
from transformers import BertTokenizerFast
from sparse_update.datamodules import DATAMODULE_REGISTER
from sparse_update.modules import PLMODULE_REGISTER
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bert_config", type=str, default="bert-base-cased")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--num_warmup_steps", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data_class = DATAMODULE_REGISTER[args.task]
    module_class = PLMODULE_REGISTER[args.task]

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_config)
    data_module = data_class(args.batch_size, args.num_workers, tokenizer)

    model = module_class(args, args.bert_config)

    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max")
    # trainer = Trainer(max_epochs=3)

    if not torch.cuda.is_available():
        args.gpus = 0

    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.max_grad_norm,
    )
    trainer.fit(model, data_module)

    trainer.test(datamodule=data_module)

    # print(metric)

    # squad_dataset = load_dataset('glue')
    # print(squad_dataset)
