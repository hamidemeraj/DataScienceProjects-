
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 
from sklearn.datasets import load_sample_image

# china = load_sample_image("flower.jpg")
china = load_sample_image("china.jpg")
 
ax = plt.axes(Xticks=[], yticks=[])
ax.imshow(china)
china.shape
print(china)

#Need to Normalize: (i-min)/(max-min) 
data = china /255.0

#Convert 3-Dimensional to 2-Dimensional 
data = data.reshape(427 * 640, 3)
data.shape


def plot_pixels(data,title, colors = None, N = 10000):
    if colors is None:
        colors = data 
    #Choose a random subset 
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (16,8))
    ax1.scatter(R, G, c = colors, marker = '.')
    ax1.set(xlabel ='Red', ylabel ='Green', xlim =(0,1), ylim=(0,1))
    ax1.set_title('Red and Green')
    fig.suptitle('Plot With 10000 Data')
    

    ax2.scatter(R, B, c = colors, marker = '.')
    ax2.set(xlabel ='Red', ylabel ='Blue', xlim =(0,1), ylim=(0,1));
    ax2.set_title('Red and Blue')
    fig.suptitle(title, size = 20 )
    plt.show()
    
plot_pixels(data=data, title = 'Scatter plot of colors')    
  
from sklearn.cluster import MiniBatchKMeans 
kmeans = MiniBatchKMeans(16)
import warnings 
warnings.simplefilter('ignore')   
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]


plot_pixels(data, title= 'Reduce color space', colors = new_colors)
china_recolored = new_colors.reshape(china.shape)
fig , ax = plt.subplots(1,2,figsize = (16,6))
fig.subplots_adjust(wspace = 0.5)
ax[0].imshow(china)
ax[0].set_title('original_image', size = 15)
ax[1].imshow(china_recolored)   
ax[1].set_title('16_color_image', size = 15)   
    

    
    
    
    