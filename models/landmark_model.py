import numpy as np
import torch
from torch import nn
from torch.optim import Adam


# i will make this class to be almost identical to the pixel model,
# some of the logic i literaly just copied from it
class Landmark_Model:
    def __init__(
        self,
        loaders,
        loss_fn=nn.CrossEntropyLoss(),
        epochs=20,
        patience=5,
        optimizer=Adam,
        lr=0.001,
        input_dim=63,
        dropout=0.3,
        num_classes=26,
        class_names=None,
        normalize=True,
        best_model_path="landmark_pretrained.pt",
    ):
        self.patience = patience
        self.lr = lr
        self.input_dim = input_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.class_names = class_names
        self.normalize = normalize
        self.best_model_path = best_model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.make_model().to(self.device)

        self.loaders = loaders
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.epochs = epochs

    def make_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.num_classes),
        )

        return model

    def print_model(self, model):
        print(model)

    # i literaly just copies thay from the pixel model and only changed var names from images to landmarks
    def eval_model(self, split="val"):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for landmarks, labels in self.loaders[split]:
                # i also just hardcode the datatyoe to fload32 for torch
                landmarks = landmarks.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                logits = self.model(landmarks)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                predictions = logits.argmax(dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += len(labels)

                accuracy = total_correct / total_samples
                loss = total_loss / total_samples

        return loss, accuracy

    def save_model(self, path=None):
        path = path or self.best_model_path
        torch.save(self.model.state_dict(), path)

    def load_model(
        self,
    ):
        path = self.best_model_path
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    # i also just copied the training loop from the pixel model
    # and changed the dataype to float32
    def train_model(self):
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acu = -1
        epochs_without_improvement = 0
        best_epoch = 0

        for epoch in range(self.epochs):
            self.model.train()

            total_loss = 0
            total_correct = 0
            total_samples = 0

            print(f"epoch: {epoch} - ", end="")
            for landmarks, labels in self.loaders["train"]:
                landmarks = landmarks.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                # clear previose optimizer gradients
                self.optimizer.zero_grad()

                # time for the forware pass
                logits = self.model(landmarks)
                loss = self.loss_fn(logits, labels)

                # backward pass
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

    # this is the only new moathod to the class, i decided to make the noramlization
    # a stand alone method insted of stuffin it into the predict method
    # (it got too ugly for my taste in the picel class and iam too lazy to refactor it tbh)
    def prepare_landmarks(self, landmarks):
        landmarks = np.asarray(landmarks, dtype=np.float32)

        # my detector gets returns a flattened array so i need to reshape it before
        # i can normalize
        landmarks = landmarks.reshape(21, 3).copy()

        # claude recoomended making the normalizarion optinal in case i change item
        # in the future (i wont) but i feel it is good practice anyways
        if self.normalize:
            # this the exact same normalization as in the loader
            # i explained it there
            landmarks = landmarks - landmarks[0]
            scale = np.abs(landmarks).max()
            if scale > 0:
                landmarks = landmarks / scale

        return landmarks.flatten()

    # again same logic as the pixel model just much simpler scine normalizaion is no
    def predict(self, landmarks):
        features = self.prepare_landmarks(landmarks)

        tonsor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tonsor)
            probabilities = torch.softmax(logits, dim=1)

            probability, predicted_class = probabilities.max(dim=1)

        return self.class_names[predicted_class.item()], probability.item()

    def test_model(self):
        _, accuracy = self.eval_model(split="test")
        print(f"test accuracy: {accuracy:.4f}")
