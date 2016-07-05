from minst import load_mnist

images,lables = load_mnist("test")
print lables[0]
import matplotlib.pyplot as plt#5,0
plt.imshow(images[0],cmap="Greys_r")
plt.show()


