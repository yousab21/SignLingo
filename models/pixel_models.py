import torch
from torch import nn as nn
from torch.optim import Adam


class CNN_Model:
    def __init__(
        self,
        loaders,
        loss_fn=nn.CrossEntropyLoss(),
        epochs=20,
        patience=5,
        optimizer=Adam,
        lr=0.001,
        grey_scale=False,
        num_classes=36,
        dropout_conv=0.2,
        dropout_fc=0.3,
    ):
        self.grey_scale = grey_scale
        self.num_classes = num_classes
        self.dropout_conv = dropout_conv
        self.dropout_fc = dropout_fc
        self.patience = patience
        # makse torch use my gpu for the training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # here i make the model in the object constructor insted of passing it as a parameter and load it into cuda
        self.model = self.make_model().to(self.device)

        self.loaders = loaders
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def make_model(self):
        if self.grey_scale:
            in_ch = 1
        else:
            in_ch = 3

        model = nn.Sequential(
            # block 1, input is 400x400 x either 3 or 1
            nn.Conv2d(in_channels=in_ch, out_channels=8, kernel_size=3, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_conv),
            # block 2, input is 200x200x8
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_conv),
            # block 3, input is 100x100x16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_conv),
            # block 4, input is 50x50x32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_conv),
            # block 5, input is 25x25x64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_conv),
            # the image is now a 12x12x128 tenosr so after flattening it bocomes a 18432 vector
            # that is way to big, here is a soltion i found
            # this AdaptiveAvgPool2d function takes an image of any sie and pools
            # it into any size of choice here 4x4 gives a nice 2048 vector,
            # this also makes the model inpedent from image size
            # fully connected section
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Dropout(self.dropout_fc),
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(),
            nn.Dropout(self.dropout_fc),
            nn.Linear(in_features=100, out_features=self.num_classes),
        )

        return model

    def print_model(self, model):
        print(model)

    def eval_model(self, split="val"):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.loaders[split]:
                # get the image and labels from my loader dict
                images = images.to(self.device)
                labels = labels.to(self.device)

                # run the image through the model and calculate the loss
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                # calculate the accuracy
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += len(labels)

                accuracy = total_correct / total_samples
                loss = total_loss / total_samples

        return loss, accuracy

    # just methods to save and load trained parameters
    def save_model(self, path="pixel_pretrained.pt"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="pixel_pretrained.pt"):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def train_model(self):
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        # i will implement early stopping with best model saving
        # start val accuracy with -1 so that the first epoch always counts as an improvment
        best_val_acu = -1
        epochs_without_improvement = 0
        best_epoch = 0

        for epoch in range(self.epochs):
            self.model.train()

            total_loss = 0
            total_correct = 0
            total_samples = 0

            print(f"epoch: {epoch} - ", end="")
            for images, labels in self.loaders["train"]:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # clear previose optimizer gradients
                self.optimizer.zero_grad()

                # time for the forware pass

                # run imagse throught the model to get logits
                logits = self.model(images)
                # calculate the loss
                loss = self.loss_fn(logits, labels)

                # backward pass :
                loss.backward()
                self.optimizer.step()

                # update tracking metrics
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += len(labels)

            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            print(
                f"train loss: {train_loss:.4f} train acc: {train_acc:.4f} - ",
                end="",
            )

            # evaluate the validation metric using my eval function
            val_loss, val_acc = self.eval_model(split="val")
            print(f"val loss: {val_loss:.4f} val acc: {val_acc:.4f}")

            # update tracing history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # early stopping
            if val_acc > best_val_acu:
                best_val_acu = val_acc
                epochs_without_improvement = 0
                best_epoch = epoch
                self.save_model()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                print(f"early stopping at epoch {epoch}")
                print(f"best epoch: {best_epoch} with val acc: {best_val_acu}")
                break

        # laod the best model in memory
        self.load_model()
        return history

    def test_model(self):
        _, accuracy = self.eval_model(split="test")
        print(f"test accuracy: {accuracy:.4f}")
