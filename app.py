from data.loaders import Pixel_Loader
from models.pixel_models import CNN_Model

Pixel_Loader = Pixel_Loader()
pixel_loaders = Pixel_Loader.get_loaders()

model = CNN_Model(loaders=pixel_loaders)

model.train_model()
