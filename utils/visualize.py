import matplotlib.pyplot as plt
import math
import collections
import numpy as np
import plotly.graph_objects as go


def format_annotation(ann):
    if isinstance(ann, str):
        return ann
    else :
        counter = collections.Counter(ann)
        return '\n'.join('{} : {}'.format(c, n) for c, n in counter.items())

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
            plt.imshow(images[i], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        else:
            plt.imshow(images[i][..., ::-1], interpolation='nearest')
        plt.title(format_annotation(annotations[i]))
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
    
    for i, hist in enumerate(histograms):
        fig.add_subplot(rows, columns, i+1)
        plt.bar(hist[1][:-1], hist[0], edgecolor="black", align="edge")
        plt.title(annotations[i])
        
    fig.tight_layout()
    plt.show()
    
    
def visualize_rgb_histograms(histograms, bins, fig_size=(20,10), fontsize=12):
    """Visualize the given histograms.

        Args:
            histogram: A list of 1D arrays.
            annotations: A list of strings to be shown above the histograms.
            fig_size: The size of the figure where the image will be shown.
            fontsize: The fontsize of the annotation.
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0,0,1,1])
    
    br1 = np.arange(len(histograms[0]))
    br2 = [x + 0.25 for x in br1]
    br3 = [x + 0.25 for x in br2]
 
    
    ax.bar(br1, histograms[0], color = 'b', width = 0.25)
    ax.bar(br2, histograms[1], color = 'g', width = 0.25)
    ax.bar(br3, histograms[2], color = 'r', width = 0.25)
    
    plt.xticks([r + 0.25 for r in range(len(histograms[0]))], br1)
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
    
def visualize_table(headers, vals):   
    fig = go.Figure(data=[go.Table(
        header=dict(values=headers,
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='center'),
        cells=dict(values=vals,
                   line_color='darkslategray',
                   fill_color='lightcyan',
                   align='center'))
    ])
    fig.show()
    
def visualize_PR_curve(pr_curves, query_names, title):
    plt.rcParams["figure.figsize"] = (20,10)

    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    for i, curve in enumerate(pr_curves):
        plt.plot(curve[1], curve[0], label=query_names[i])
        
    plt.legend()
    plt.show()
    
def visualize_PR_interpolated(precision, interpolated_precision, interval_precision, recall, title):
    fig, ax = plt.subplots(1,1)
    
    ax.plot(recall, precision, '--b')
    ax.step(recall, interpolated_precision, '-r')
    ax.scatter(np.arange(1.1, step=0.1), interval_precision)
    
    ax.set_xlim([0,1.0])
    ax.set_ylim([0,1.0])
    
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    
    plt.show()
