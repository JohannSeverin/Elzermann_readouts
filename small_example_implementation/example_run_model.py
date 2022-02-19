import numpy as np
# import os
from tensorflow.keras.models import load_model

data_path = "traces_high.txt"

model = load_model("1D_conv")
data  = np.genfromtxt(data_path, delimiter = " ")

predictions = model(data).numpy()


np.savetxt(data_path[:-4] + "_predictions.txt", predictions)





