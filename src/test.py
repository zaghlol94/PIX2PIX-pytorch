import numpy as np
import config
import os
from PIL import Image

image = np.array(Image.open("data/val/1.jpg"))
input_image = Image.fromarray(image[:, :600, :])
target_image = Image.fromarray(image[:, 600:, :])

input_image.save("x.png")
target_image.save("y.png")
