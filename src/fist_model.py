from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import cnn_learner
from fastai.vision.augment import Resize

from torchvision.models import resnet34
#from fastai.callback.all import *
from fastai.callback.schedule import fine_tune
from fastai.callback.progress import ProgressCallback

# data
path = untar_data(URLs.PETS)
files = get_image_files(path / "images")

# data loader
data = ImageDataLoaders.from_name_func(path, files, lambda x: x[0].isupper(), item_tfms=Resize(224))

# learning
learn = cnn_learner(data, resnet34)

# tune
learn.fine_tune(1)
