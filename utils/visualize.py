import matplotlib.pyplot as plt
import math


def visualize_images(images, annotations, fig_size=(20,10), fontsize=12, gray=False):
    """Visualize the given images in a Mx5 table.

        Args:
            images: A list of raw images to be shown.
            annotations: A list of strings to be shown above the images.
            fig_size: The size of the figure where the image will be shown.
            fontsize: The fontsize of the annotation.
            gray: A Bool indicating if the images are in grayscale format.
    """
    plt.rcParams.update({'font.size': fontsize})
    num_images = len(images)
    columns = 5
    rows = math.ceil(num_images/columns)
    fig = plt.figure(figsize=fig_size)
    
   
    for i in range(num_images):
        sub = fig.add_subplot(rows, columns, i+1)
        if gray == True:
            plt.imshow(images[i][..., ::-1], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        else:
            plt.imshow(images[i][..., ::-1], interpolation='nearest')
        plt.title(annotations[i])
        plt.axis('off')
    plt.show()
    
    
def visualize_histograms(histograms, annotations, fig_size=(20,10), fontsize=12):
    """Visualize the given histograms.

        Args:
            histogram: A list of 1D arrays.
            annotations: A list of strings to be shown above the histograms.
            fig_size: The size of the figure where the image will be shown.
            fontsize: The fontsize of the annotation.
    """
    plt.rcParams.update({'font.size': fontsize})
    num_histogram = len(histograms)
    columns = 5
    rows = math.ceil(num_histogram/columns)
    fig = plt.figure(figsize=fig_size)
    
    for i in range(num_histogram):
        fig.add_subplot(rows, columns, i+1)
        plt.bar(histograms[i][1][:-1], histograms[i][0], edgecolor="black", align="edge")
        plt.title(annotations[i])
        
    fig.tight_layout()
    plt.show()

    
def visualize_histogram(histogram, annotation, fig_size=(20,10), fontsize=12):
    """Visualize the given histogram.

        Args:
            histogram: A 1D array.
            annotation: A string to be shown above the histogram.
            fig_size: The size of the figure where the image will be shown.
            fontsize: The fontsize of the annotation.
    """
    plt.rcParams.update({'font.size': fontsize})
    fig = plt.figure(figsize=fig_size)
    plt.bar(histogram[1][:-1], histogram[0], edgecolor="black", align="edge")
    plt.title(annotation)
    plt.xlabel('bins')
    plt.ylabel('probability')
    plt.show()