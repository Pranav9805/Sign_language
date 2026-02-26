import pandas as pd
import numpy as np
from PIL import Image

# Load the Sign Language MNIST test CSV
test_df = pd.read_csv('data/sign_mnist_test.csv')

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
classes.remove('J')
classes.remove('Z')

# Generate and save first 10 test images
for index in range(10):
    row = test_df.iloc[index]
    pixels = row[1:].values.astype(np.uint8)
    image_array = pixels.reshape(28, 28)
    img = Image.fromarray(image_array, mode='L')

    filename = f'test_image_{index}_{classes[row["label"]]}.png'
    img.save(filename)
    print(f'Saved {filename}')
