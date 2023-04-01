import lightning as pl
import pandas as pd
import torch
import torchmetrics
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset


class LitModel(pl.LightningModule):
    def __init__(self, user_count, item_count, embedding_size, num_classes):
        super().__init__()
        self.user_embedding = nn.Embedding(user_count, embedding_size)
        self.item_embedding = nn.Embedding(item_count, embedding_size)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, num_classes),
        )
        self.loss_fn = CrossEntropyLoss()

        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task="binary", num_classes=num_classes)
        self.precision = torchmetrics.Precision(task="binary", num_classes=num_classes)
        self.recall = torchmetrics.Recall(task="binary", num_classes=num_classes)

        self.training_step_outputs = {
            "pred": torch.tensor([]),
            "true": torch.tensor([]),
        }

    def forward(self, x):
        user_ids, item_ids = x
        return self._predict(user_ids, item_ids)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        pred = self.training_step_outputs["pred"]
        true = self.training_step_outputs["true"]
        f1_score = self.f1(pred, true)
        acc_score = self.acc(pred, true)
        precision_score = self.precision(pred, true)
        recall_score = self.recall(pred, true)
        scores = {
            "f1": f1_score,
            "acc": acc_score,
            "precision": precision_score,
            "recall": recall_score,
        }
        for key, value in scores.items():
            self.log(key, value)
        print(scores)
        self.training_step_outputs["pred"] = torch.Tensor([])
        self.training_step_outputs["true"] = torch.Tensor([])

    def _predict(self, user_ids, item_ids):
        users = self.user_embedding(user_ids)
        items = self.item_embedding(item_ids)
        cat = torch.cat((users, items), dim=1)
        logits = self.classifier(cat)
        return logits

    def _common_step(self, batch, batch_idx, stage):
        user_ids, item_ids, labels = batch
        logits = self._predict(user_ids, item_ids)
        if stage == "train":
            self.training_step_outputs["pred"] = torch.cat(
                (self.training_step_outputs["pred"], logits.argmax(-1))
            )
            self.training_step_outputs["true"] = torch.cat(
                (self.training_step_outputs["true"], labels)
            )

        loss = self.loss_fn(logits, labels)
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    user_key = "userId"
    item_key = "itemId"
    rating_key = "ratings"
    path = "~/develop/dataset/movielens/ml-100k/"
    df = pd.read_csv(f"{path}/u.data", delimiter="\t")
    df.columns = [user_key, item_key, rating_key, "timestamp"]
    df = df.drop(columns=["timestamp"])
    df["ratings"] = (df["ratings"] > 2).astype("int")
    train_target = torch.tensor(df["ratings"].values.astype("int"))
    user_ids = torch.tensor(df[user_key].values.astype("int"))
    item_ids = torch.tensor(df[item_key].values.astype("int"))
    train_tensor = TensorDataset(user_ids, item_ids, train_target)
    train_loader = DataLoader(dataset=train_tensor, batch_size=64, shuffle=False)
    user_count = df[user_key].max() + 1
    item_count = df[item_key].max() + 1
    model = LitModel(
        user_count=user_count, item_count=item_count, embedding_size=64, num_classes=2
    )
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer(max_epochs=20, logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader)
    trainer.test(model, dataloaders=train_loader)


if __name__ == "__main__":
    main()
