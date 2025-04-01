import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_process import load_dataset


def visualize_images(dataset):
    """
    Visualize images by matplotlib

    Parameters:
    dataset - a list of PIL Image and label pairs
    """
    # config plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))  # 2行5列布局
    axes = axes.ravel()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # iterate images
    for _, (ax, (img, labels)) in enumerate(zip(axes, dataset)):
        # show image
        ax.imshow(img)
        for label in labels:
            # create a polygon
            quadrilateral = patches.Polygon(
                label['position'],
                closed=True,  # closed polygon
                linewidth=1,  # line width of edge
                edgecolor='blue' if label['class'] == 'plane' else 'red',  # edge color
                alpha=0.4,  # transparency
                facecolor='none',  # filled color
            )
            # draw a polygon
            ax.add_patch(quadrilateral)
        # close axis
        ax.axis('off')

    # auto layout
    plt.tight_layout()
    plt.show()


# test visualize_images
def test_visualize_images():
    dataset = load_dataset(top_num=9, include_label=True)
    visualize_images(dataset)


if __name__ == "__main__":
    test_visualize_images()
