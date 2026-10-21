from loaders import Pixel_Loader, Landmark_Loader

pixel_colored = Pixel_Loader(grey_scale=False)
pixel_greyscale = Pixel_Loader(grey_scale=True)
landmark = Landmark_Loader()

pixel_loader_coloered = pixel_colored.get_loaders()
pixel_loader_greyscaled = pixel_greyscale.get_loaders()
landmark_loader = landmark.get_loaders()


print("Pixel data colored")
images, image_labels = next(iter(pixel_loader_coloered["train"]))
print(images.shape, image_labels.shape)
print()
print("Pixel data greyscaled")
images, image_labels = next(iter(pixel_loader_greyscaled["train"]))
print(images.shape, image_labels.shape)
print()

print("Landmark data")
features, landmark_labels = next(iter(landmark_loader["train"]))
print(features.shape, landmark_labels.shape)
