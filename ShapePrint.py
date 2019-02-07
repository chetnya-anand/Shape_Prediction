
# coding: utf-8

# In[2]:


from TestTrainAndModel import model


# In[9]:


from DataGeneration import *


# In[13]:


from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
shape_input  = input('Enter the Figure Name\n')
# Create a random test image
img = create_image ((128,128), shape_input)

# The model expects a batch of images as input, so we'll create an array of 1 image
imgfeatures = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

# We need to format the input to match the training data
# The generator loaded the values as floating point numbers
# and normalized the pixel values, so...
imgfeatures = imgfeatures.astype('float32')
imgfeatures /= 255

# Use the classifier to predict the class
class_probabilities = model.predict(imgfeatures)
# Find the class predictions with the highest predicted probability
class_idx = np.argmax(class_probabilities, axis=1)
plt.imshow (create_image((128,128),(classnames[int(class_idx[0])])))
plt.show()

