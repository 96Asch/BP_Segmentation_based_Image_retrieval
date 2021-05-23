import numpy as np



def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

        Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

        Args:
            label: A 2D array with integer type, storing the segmentation label.

        Returns:
            result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

        Raises:
            ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expected 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def extract_segment_from_image(image, segment):

    """Extracts a subimage according to the segmented area
            Args:
            image: A 2D array of RGB integer values that describe
            the original (resized) image.

            segment: A 2D array of integer values that describe the
            segmented areas of the image.

        Returns:
            return segment_out: A 3D array of RGB integer values that describe
            each of the extracted segments from the original image.

        Raises:
            ValueError: If image is not of rank 3 or if segment is not of rank 2
            RunTimeError: If the 2d dimensions of the segment and
            the original image are not the same
    """
    if image.ndim != 3:
        raise ValueError('Expected 2-D RGB image')
    if segment.ndim != 2:
        raise ValueError('Expected 2-D segment')

    if image.shape[:2] != segment.shape:
        raise RunTimeError('Image and segment are not same size')

    unique_segments = np.unique(segment)
    print(unique_segments)
    segment_out_shape = (len(unique_segments), ) + image.shape
    segment_out = np.zeros(segment_out_shape, dtype=np.uint8)
    dim = image.shape[:2]
    percentages_filled = np.zeros(len(unique_segments), dtype=np.float)
    
    for h in range(len(unique_segments)):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if segment[i][j] == unique_segments[h]:
                    segment_out[h][i][j] = image[i][j]
                    percentages_filled[h] += 1                  
                    
    print("before: ", percentages_filled)
    print("dim: ", dim)
    percentages_filled /= (dim[0] * dim[1])
    print("after: ", percentages_filled)
    print("sum: ", np.sum(percentages_filled))
    return segment_out, percentages_filled
