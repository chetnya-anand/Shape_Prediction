
# coding: utf-8

# In[7]:


import numpy as np
import sklearn
from skimage.measure import block_reduce
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFilter
import tensorflow
from tensorflow.keras import backend as K
get_ipython().run_line_magic('matplotlib', 'inline')
def create_image (size, shape):
    from random import randint
    import numpy as np
    from PIL import Image, ImageDraw
    
    xy1 = randint(10,40)
    xy2 = randint(60,100)
    col = (randint(0,200), randint(0,200), randint(0,200))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    if shape == 'circle':
        draw.ellipse([(xy1,xy1), (xy2,xy2)], fill=col)
    elif shape == 'triangle':
        draw.polygon([(xy1,xy1), (xy2,xy2), (xy2,xy1)], fill=col)
    else: # square
        draw.rectangle([(xy1,xy1), (xy2,xy2)], fill=col)
    del draw
    
    return np.array(img)
# Create a 128 x 128 pixel image (Let's use a square)
img = Image.fromarray(create_image((128,128), 'square'))

# Now let's generate some feature extraction layers
layers = []

# Define filter kernels - we'll use two filters
kernel_size = (3,3) # kernels are 3 x 3
kernel_1 = (1, 0, -100,
            1, 0, -100,
            1, 0, -100) # Mask for first filter
kernel_2 = (-200, 0, 0,
            0, -200, 0,
            -0, 0, -200) # Mask for second filter

# Define kernel size for pooling
pool_size = (2,2,1) # Pool filter is 2 x 2 pixels for each channel (so image size will half with each pool)
this_layer = []
# Apply each filter to the original image - this generates a layer with two filtered images
this_layer.append(np.array(img.filter(ImageFilter.Kernel(kernel_size, kernel_1))))
this_layer.append(np.array(img.filter(ImageFilter.Kernel(kernel_size, kernel_2))))
layers.append(this_layer)
# Add a Pooling layer - pool each image in the previous layer by only using the maximum value in each 2x2 area
this_layer = []
for i in layers[len(layers)-1]:
    # np.maximum implements a ReLU activation function so all pixel values are >=0
    this_layer.append(np.maximum(block_reduce(i, pool_size, np.max), 0))
layers.append(this_layer)

# Add a second convolutional layer - generates a new layer with 4 images (2 filters applied to 2 images in the previous layer)
this_layer = []
for i in layers[len(layers)-1]:
    this_layer.append(np.array(Image.fromarray(i).filter(ImageFilter.Kernel(kernel_size, kernel_1))))
    this_layer.append(np.array(Image.fromarray(i).filter(ImageFilter.Kernel(kernel_size, kernel_2))))
layers.append(this_layer)

# Add a second Pooling layer - pool each image in the previous layer
this_layer = []
for i in layers[len(layers)-1]:
    # np.maximum implements a ReLU activation function so all pixel values are >=0
    this_layer.append(np.maximum(block_reduce(i, pool_size, np.max), 0))
layers.append(this_layer)
# Set up a grid to plot the images in each layer
#fig = plt.figure(figsize=(16, 24))
#rows = len(layers) + 1
#columns = len(layers[len(layers)-1])
#row = 0
#image_no = 1

# Plot the original image as layer 1
#a=fig.add_subplot(rows,columns,image_no)
#imgplot = plt.imshow(img)
#a.set_title('Original')

# Plot the convolved and pooled layers
#for layer in layers:
#   row += 1
#    image_no = row * columns
#    for image in layer:
#       image_no += 1
#        a=fig.add_subplot(rows,columns,image_no)
#        imgplot = plt.imshow(image)
#        a.set_title('Layer ' + str(row))

# function to create a dataset of images
def generate_image_data (classes, size, cases, img_dir):
    import os, shutil
    from PIL import Image
    
    if os.path.exists(img_dir):
        replace_folder = input("Image folder already exists. Enter Y to replace it (this can take a while!). \n")
        if replace_folder == "Y":
            print("Deleting old images...")
            shutil.rmtree(img_dir)
        else:
            return # Quit - no need to replace existing images
    os.makedirs(img_dir)
    print("Generating new images...")
    i = 0
    while(i < (cases - 1) / len(classes)):
        if (i%25 == 0):
            print("Progress:{:.0%}".format((i*len(classes))/cases))
        i += 1
        for classname in classes:
            img = Image.fromarray(create_image(size, classname))
            saveFolder = os.path.join(img_dir,classname)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            imgFileName = os.path.join(saveFolder, classname + str(i) + '.jpg')
            try:
                img.save(imgFileName)
            except:
                try:
                    # Retry (resource constraints in Azure notebooks can cause occassional disk access errors)
                    img.save(imgFileName)
                except:
                    # We gave it a shot - time to move on with our lives
                    print("Error saving image", imgFileName)
            
# Our classes will be circles, squares, and triangles
classnames = ['circle', 'square', 'triangle']

# All images will be 128x128 pixels
img_size = (128,128)

# We'll store the images in a folder named 'shapes'
folder_name = 'shapes'

# Generate 1200 random images.
generate_image_data(classnames, img_size, 1200, folder_name)

print("Image files ready in %s folder!" % folder_name)

