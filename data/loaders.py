from pathlib import Path  # for path manip
from PIL import (
    Image,
)  # this is a libiray called pillow it adds alot of usfull stuff  for laoding images localy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2

# normally python looks to dir relative to the caller file (in my case app.py)
# not the files that actually use the dir,
# the path lib has way to "anchor" the data path to the files that use it
# path.resolve() will return the absolute path
# .parent drops the filename, so paths built from it ignore the working directory.
DATA_ROOT = Path(__file__).resolve().parent


# i noticed that this logic is gonna repaet for both loader classes so i pulled it into a base class
#
class Base_Loader:
    # some guy on reddit reccomended this, the idea is to make each class only accept a set file extensions
    # that way, if the data dirs ever contain other files, they will be ignored
    extensions = set()

    def __init__(self, data_dir, batch_size, val_precent, test_percent, seed):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_precent = val_precent
        self.test_percent = test_percent
        self.seed = seed

        # the class names list will by load meathods
        self.class_names = []

        # this is a neat idea that claude reccomendede to me:
        # this meathods will scan every syb folder in the data dir and
        # return two lists of the path of each image and the class name for thae image at that path

    def get_paths_and_labels(self):
        if not self.extensions:
            raise NotImplementedError("Child classes must define 'extensions'.")

        # in linux naming blobs represent both files and folders, iterdir() gets all of them recursivly
        all_blobs = list(self.data_dir.iterdir())

        # if the blob is a dir then it is a class, get every di blob in the data dir
        class_dirs = []
        for blob in all_blobs:
            if blob.is_dir():
                class_dirs.append(blob)

        # get the class names based on the dir they are in
        class_names = []
        for dir in class_dirs:
            class_names.append(dir.name)

        # apperently path.iterdir() returns blobs in random order,
        # so i order them so that class numbers are the same between each run
        class_names.sort()

        # save the class names in the class instance
        self.class_names = class_names

        # the two lists to return, where each path[i] is of class[i]
        paths = []
        labels = []

        # iterate over each class
        for label, class_name in enumerate(self.class_names):
            # get the pth
            class_dir = self.data_dir / class_name

            # get the files
            files = list(class_dir.iterdir())

            # sort in place so every run is the same
            files.sort()

            for file in files:
                # read the file's extension
                extension = file.suffix

                # linux treats .png and .PNG as the same file, so this code should to
                extension = extension.lower()

                # keep the file only if its type is allowed in the sepicific loader
                if extension in self.extensions:
                    paths.append(file)
                    labels.append(label)

        # conver the labls list to numpy a array
        labels_array = np.array(labels)

        return paths, labels_array

    # this meathod just gets the class name for a passed in label,
    # making use of the class names attribute i made
    def get_labele_class_name(self, label):
        return self.class_names[label]

    def split_data(self, X, y):
        # for conviniece sake to keep the data size dynamic
        # i will just use sklearn train_test_split insted of hard coding slices

        train_and_test_split = train_test_split(
            X,
            y,
            test_size=self.test_percent,
            stratify=y,
            random_state=self.seed,
        )
        # ater this split i have a ready test set, and what would become the train and val set

        # split in to the train and val combo and the test set
        X_rest, X_test, y_rest, y_test = train_and_test_split

        # i now split the rest of the data other then the test set into the train and val set
        trian_and_val_split = train_test_split(
            X_rest,
            y_rest,
            test_size=self.val_precent,
            stratify=y_rest,
            random_state=self.seed,
        )

        X_train, X_val, y_train, y_val = trian_and_val_split

        train_pair = (X_train, y_train)
        val_pair = (X_val, y_val)
        test_pair = (X_test, y_test)

        # after i pair them i group them into a dict to return it
        splits = {}
        splits["train"] = train_pair
        splits["val"] = val_pair
        splits["test"] = test_pair

        return splits

    # now i actually make the loaders
    def make_loaders(self, splits):
        loaders = {}

        # lets load the numpy arrays into actual torch tensor datasets
        for name, pair in splits.items():
            X, y = pair

            # turn the numpy arrays into torch tensors
            X_tensor = torch.from_numpy(X)
            y_tensor = torch.from_numpy(y)

            # make the tensor dataset
            dataset = TensorDataset(X_tensor, y_tensor)

            # create the data laoder and make it shuffle only the train set
            # (i felt reallt smart writing this inline condition XD)
            loaders[name] = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=(name == "train")
            )

        return loaders

    def load_features(self, paths):
        # this method is the one that must be overwitten by each data subclass
        # it should take a list of file paths and get the actual features
        # whehever they may be, and turn said features in a float32 numpt array
        raise NotImplementedError("subclass must implement this method")

    # this is the method i will actually call when useing the interface i have
    # been designign for the past 3 hours, it just runs the whole pipeline
    # and returns the ready to use laoders, (assuming load_features is implement)
    def get_loaders(self):
        paths, labels = self.get_paths_and_labels()
        X = self.load_features(paths)
        splits = self.split_data(X, labels)
        loaders = self.make_loaders(splits)
        return loaders


class Pixel_Loader(Base_Loader):
    # set the allowed format for pixel data (in my cass all of the asl data are .jpeg)
    extensions = {".jpg", ".jpeg", ".png"}

    # i call the base constructor becaude all of the logic are from the base class and usses these parameters
    # after testing the model i found out it is highly relaiant on the camera prespective and recommendation
    # so i need to implement some sort of data augmentation
    # i dont want to mess with other laoder classes so i will keep all changes in this class
    def __init__(
        self,
        data_dir=DATA_ROOT / "pixel-data",
        image_size=(400, 400),
        batch_size=32,
        val_percent=0.15,
        test_percent=0.15,
        seed=42,
        # i dont really know if grey scalling the images will help so i will add
        # an option for it rather than hardcoding it
        grey_scale=False,
        # flag to augment the data
        augment=True,
        # number of augmented versions added per original image in every training batch
        # the batch now holds batch_size * (1 + augment_copies) images
        augment_copies=1,
    ):
        super().__init__(data_dir, batch_size, val_percent, test_percent, seed)

        self.image_size = image_size
        self.grey_scale = grey_scale
        self.augment = augment
        self.augment_copies = augment_copies

        self.transform = v2.Compose(
            [
                # mirror the images so the model works with left and right hands
                v2.RandomHorizontalFlip(p=0.5),
                # rotation up to 20 degrees either way
                # shift of up to 10% of the image size
                # zoom between 80% and 120%
                v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                # this is supposed to simulate viewing the frame from an angle
                v2.RandomPerspective(distortion_scale=0.2, p=0.5),
                # change the brightness and contrast
                v2.ColorJitter(brightness=0.3, contrast=0.3),
            ]
        )

    # impelemnt loading image data set
    def load_features(self, paths):
        images = []

        if self.grey_scale:
            mode = "L"
        else:
            mode = "RGB"

        for path in paths:
            # use pil to open teh image
            image = Image.open(path)
            # convert to rgb
            image = image.convert(mode)
            # resize so all images have the same size
            image = image.resize(self.image_size)
            # the np asarray main feature is it doent copy the data, so it is a lot faster (claude recommendation)
            image = np.asarray(image, dtype=np.float32)
            # normmalize
            image = image / 255

            images.append(image)

        images = np.array(images)

        # stake all imahes into one array, axis 0 is the num of images
        # right now the shape should be (num_images, hieght (400), width (400), channels (3))
        X = np.stack(images)

        # a proplem with supporting greyscaled data is tha the imaages lose one dimention (channels)
        # so i will add one to the end to make it (num_images, hieght, width, channels)
        # the mothod np.expand_dims() can do that, the axis = -1 means the one will be added to the end
        if self.grey_scale:
            X = np.expand_dims(X, axis=-1)

        # to make it work with torch conv layrs i will reorde the dimentions to be (num_images, channels, hieght, width)
        # this is just equivilant to torch.permute(0, 3, 1, 2), only on the numpy side
        X = X.transpose(0, 3, 1, 2)

        return X

    # this build the training batch from both clean and augmented images
    def augment_batch(self, samples):
        # get the images labels from the tensors into paralla arrays
        images = [image for image, _ in samples]
        labels = [label for _, label in samples]

        # turn the image into a noraml list
        all_images = list(images)

        # look at this buiteiful inline loop and no i didnt come up with it
        # you DONT want to see how i did this before papa claude put some magic in my brain XD
        # the loop runs for how much extra data-sized copies you want i just set it 1 so it dobles the data
        # and it runs transform on each image by its own so that each one gets a different augmentation
        # that is why i turned them into a pure list
        for _ in range(self.augment_copies):
            all_images.extend(self.transform(image) for image in images)

        # as the imaes are just repeated in the same order just with augmentation
        # i can just repeate the lables and they will match, isntthis is just beuetiful
        all_labels = labels * (1 + self.augment_copies)

        # turn it back to a tensor and return it

        return torch.stack(all_images), torch.stack(all_labels)

    # i also need to overwirte the make_loaders method to add the augmentation

    # this in my opinion is the most beuetuful piece of code in the whole project
    def make_loaders(self, splits):
        # this makes the three loaders normally in with the implementation in the parent class
        loaders = super().make_loaders(splits)

        # if the augment flag is tru apply the augmentation, here is hwo i do that
        if self.augment:
            # remember how we already created the three loaders using the parent's implementation of the method
            # i only replace the train loader which is the only one that needs to be augmented,
            # the whole reson this is possible is because of that collate_fn parameter,
            # what the functio is suposed to do in general is to asseble tensors into a batch
            # torch has a defual impelemntation for it which i can replace with my own by passing in a function
            # by defualt it just stacks the exsisting tensors into one big tensor and calles it a batch
            # but i replace it with my augment_batch function, wich adds augmenated samples to the exsisting ones
            # before you ask this also was not my idea, thatnk you papa claude
            loaders["train"] = DataLoader(
                loaders["train"].dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.augment_batch,
            )
        return loaders


class Landmark_Loader(Base_Loader):
    """Provides landmark DataLoaders with batches shaped (batch, 63) for a linear model."""

    extensions = {".npy"}

    def __init__(
        self,
        data_dir=DATA_ROOT / "landmark-data",
        normalize=True,
        batch_size=32,
        val_size=0.15,
        test_size=0.15,
        seed=42,
    ):
        super().__init__(data_dir, batch_size, val_size, test_size, seed)

        self.normalize = normalize

    def load_features(self, paths):
        samples = []

        for path in paths:
            # E=each file contains a (21, 3) array, 21 joints with x, y, z.
            landmarks = np.load(path).astype(np.float32)

            # common normalization for hand landmark data i found online
            if self.normalize:
                # shift all point so that the wrist (joint 0) is the origin which
                # which makes the hand position irrelevant
                landmarks = landmarks - landmarks[0]

                # Divide by the largest coordinate magnitude
                # which makes the hand size and distance to the camera irrelevant
                scale = np.abs(landmarks).max()
                if scale > 0:
                    landmarks = landmarks / scale

            # flatten the samples
            sample = landmarks.flatten()
            samples.append(sample)

        return np.stack(samples)
