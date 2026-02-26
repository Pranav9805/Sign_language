import pandas as pd
import numpy as np
from PIL import Image

# Load the Sign Language MNIST test CSV
test_df = pd.read_csv('data/sign_mnist_test.csv')

# Select the first row (or any row number you want to test)
row = test_df.iloc[0]

# Print true label index and corresponding letter
print(f"True label index: {row['label']}")

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
classes.remove('J')
classes.remove('Z')

print(f"Corresponding letter: {classes[row['label']]}")

# Extract pixel values (all columns except 'label')
pixels = row[1:].values.astype(np.uint8)

# Reshape to 28x28 image (grayscale)
image_array = pixels.reshape(28, 28)

# Convert to PIL Image with grayscale mode
img = Image.fromarray(image_array, mode='L')

# Save the image as PNG
img.save('test_image.png')

print("Test image saved as test_image.png")
