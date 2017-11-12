import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=10, ncols=10)

#plot all images and add labels
class_number = 0
image_number = 0
for image_class in range(10):
    axes[0, image_class].set_title(image_class)
    axes[image_number, class_number].imshow((image)/255)
plt.show()