from tensorflow.keras.models import load_model
import numpy as np


model_file_path = "my_model.h5"
loaded_model = load_model(model_file_path)

x = np.random.rand(10, 10, 3)

y = loaded_model.predict(x)
y_pred = np.argmax(y, axis=1)
print(y_pred)