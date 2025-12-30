'''import matplotlib.pyplot as plt
from PIL import Image

def plot_square_on_image(image_path, x_min, y_min, x_max, y_max):
    """
    Plots a square on an image based on the given normalized coordinates (0-1 range).

    :param image_path: Path to the image file.
    :param x_min: Minimum x-coordinate of the square (normalized, 0-1).
    :param y_min: Minimum y-coordinate of the square (normalized, 0-1).
    :param x_max: Maximum x-coordinate of the square (normalized, 0-1).
    :param y_max: Maximum y-coordinate of the square (normalized, 0-1).
    """
    # Load the image
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Convert normalized coordinates to pixel coordinates
    x_min_pixel = x_min * image_width
    y_min_pixel = y_min * image_height
    x_max_pixel = x_max * image_width
    y_max_pixel = y_max * image_height

    # Create a matplotlib figure
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot the square as a rectangle
    width = x_max_pixel - x_min_pixel
    height = y_max_pixel - y_min_pixel

    # Add rectangle to the plot
    rectangle = plt.Rectangle((x_min_pixel, y_min_pixel), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rectangle)

    # Show the plot
    plt.axis('off')  # Turn off the axes
    plt.show()

plot_square_on_image('frame_0223.jpg', x_min=0.33, y_min=0.34, x_max=0.56, y_max=0.57)'''

import matplotlib.pyplot as plt
from PIL import Image

def plot_square_on_image(image_path, center_x, center_y, width, height):
    """
    Plots a bounding box on an image based on YOLO format normalized coordinates (0-1 range).

    :param image_path: Path to the image file.
    :param center_x: Center x-coordinate of the box (normalized, 0-1).
    :param center_y: Center y-coordinate of the box (normalized, 0-1).
    :param width: Width of the box (normalized, 0-1).
    :param height: Height of the box (normalized, 0-1).
    """
    # Load the image
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Convert normalized YOLO coords to pixel coordinates
    x_min_pixel = (center_x - width / 2) * image_width
    y_min_pixel = (center_y - height / 2) * image_height
    box_width_pixel = width * image_width
    box_height_pixel = height * image_height

    # Create a matplotlib figure
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot the rectangle
    rectangle = plt.Rectangle(
        (x_min_pixel, y_min_pixel),
        box_width_pixel,
        box_height_pixel,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rectangle)

    # Show the plot
    plt.axis('off')
    plt.show()


# Beispiel: YOLO-Koordinaten
plot_square_on_image('frame_0222_png.rf.35b043f351cf1fb6fda15a5ae8253c88.jpg', center_x=0.56, center_y=0.52, width=0.04, height=0.03)
