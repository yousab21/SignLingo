from loaders import Pixel_Loader, Landmark_Loader

pixel_augmented = Pixel_Loader(augment=True)
pixel_not_augmente = Pixel_Loader(augment=False)
landmark = Landmark_Loader()

pixel_loader_augmented = pixel_augmented.get_loaders()
pixel_loader_not_augmented = pixel_not_augmente.get_loaders()
landmark_loader = landmark.get_loaders()


print("Pixel data not augmented")
images, image_labels = next(iter(pixel_loader_not_augmented["train"]))
print(images.shape, image_labels.shape)
print()

print("Pixel data augmented")
images, image_labels = next(iter(pixel_loader_augmented["train"]))
print(images.shape, image_labels.shape)
print()

print("Landmark data")
features, landmark_labels = next(iter(landmark_loader["train"]))
print(features.shape, landmark_labels.shape)
