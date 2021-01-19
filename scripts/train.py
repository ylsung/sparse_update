import os
from transformers import BertTokenizerFast
from sparse_update.datamodules import DATAMODULE_REGISTER
from sparse_update.modules import PLMODULE_REGISTER
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # print(list_datasets())

    # metric = load_metric("glue", "sst2")

    task = "sst2"
    num_workers = 8
    batch_size = 32
    bert_config = "bert-base-cased"

    data_class = DATAMODULE_REGISTER[task]
    module_class = PLMODULE_REGISTER[task]

    tokenizer = BertTokenizerFast.from_pretrained(bert_config)
    data_module = data_class(batch_size, num_workers, tokenizer)

    model = module_class(bert_config, tokenizer)

    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max")
    # trainer = Trainer(max_epochs=3)
    trainer = Trainer(max_epochs=3, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)

    trainer.test(datamodule=data_module)

    # print(metric)

    # squad_dataset = load_dataset('glue')
    # print(squad_dataset)
