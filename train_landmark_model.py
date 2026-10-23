from data.loaders import Landmark_Loader
from models.landmark_model import Landmark_Model

landmark_loader = Landmark_Loader()
landmark_loaders = landmark_loader.get_loaders()

model = Landmark_Model(loaders=landmark_loaders, dropout=0.5)

model.train_model()

model.test_model()
