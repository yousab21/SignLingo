from loaders import Pixel_Loader, Landmark_Loader

pixel = Pixel_Loader()
landmark = Landmark_Loader()

pixel_loaders = pixel.get_loaders()
landmark_loaders = landmark.get_loaders()


print("Pixel data")
images, image_labels = next(iter(pixel_loaders["train"]))
print(images.shape, image_labels.shape)

print("\nLandmark data")
features, landmark_labels = next(iter(landmark_loaders["train"]))
print(features.shape, landmark_labels.shape)
