from pytorch_lightning import Callback


class TrackLinearAccAtMinLossCallback(Callback):
    """
    Tracks linear/SVM accuracy at the epoch with minimum validation loss.

    Safe for runs where SVM validation is disabled.
    If val_loss or svm_val_acc_modelnet40 is missing, it simply does nothing.
    """

    def __init__(self):
        super().__init__()
        self.min_val_loss = float("inf")
        self.acc_val_acc_modelnet40_at_min_loss = None

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        val_loss = metrics.get("val_loss")
        svm_acc = metrics.get("svm_val_acc_modelnet40")

        # In no-SVM pretraining configs, svm_acc will be missing.
        # In limited/debug validation, val_loss may also be missing.
        if val_loss is None or svm_acc is None:
            return

        val_loss = val_loss.item()
        svm_acc = svm_acc.item()

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.acc_val_acc_modelnet40_at_min_loss = svm_acc

        pl_module.log(
            "svm_val_acc_modelnet40_at_min_loss",
            self.acc_val_acc_modelnet40_at_min_loss,
        )
