import lightning as pl

DEFAULT_TRAIN_DATALOADER_KWARGS = {
    "shuffle": True,
    "batch_size": 64,
    "num_workers": 2,
    "pin_memory": True,
}
DEFAULT_VAL_DATALOADER_KWARGS = {
    "batch_size": 64,
    "num_workers": 2,
}
DEFAULT_TEST_DATALOADER_KWARGS = {
    "batch_size": 64,
    "num_workers": 2,
}


class BaseDataset(pl.LightningDataModule):
    pass
