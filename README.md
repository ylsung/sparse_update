# Sparse Update
Sparse updates is a technique that train neural networks only updating few parameters.

## Code format
We follow the `black` format.

## Usage

### Installation
It's recommended to install in conda or virtual enviroment.

```
cd sparse_update
pip install -e .
```

### Arguments
Please refer to `scripts/train.py`

### Train

```
python scripts/train.py
```

The script will also generate predictions on test data (save as `[Dataset].tsv`), which is used to submit to the Glue server for testing.


## Todo
- [ ] Support sparsed update ([diff pruning](https://openreview.net/pdf?id=E4PK0rg2eP))
- [ ] Support all Glue tasks (Add corresponding classes in `datamodules.py` and `modules.py`)
- [ ] Support pretraining tasks


## Resources
The repo base on the following projects <br/>
- [Transformers from Hugging Face](https://github.com/huggingface/transformers)