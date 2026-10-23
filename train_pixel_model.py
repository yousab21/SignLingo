from data.loaders import Pixel_Loader
from models.pixel_models import CNN_Model

pixel_loader = Pixel_Loader(augment=True)
pixel_loaders = pixel_loader.get_loaders()

model = CNN_Model(
    loaders=pixel_loaders,
    lr=0.001,
    num_classes=len(pixel_loader.class_names),
    dropout_conv=0.3,
    dropout_fc=0.3,
)

model.train_model()

model.test_model()
