import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__=="__main__":
    image = cv2.imread('picture/4.png')
    cv2.imshow('origin', image)
    cv2.waitKey()

    # Convert the original image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Black and white inversion operation to facilitate model identification
    reverse = 255 - gray

    # Threshold processing operation
    # ret1, binary_otsu = cv2.threshold(reverse,180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret1, binary_otsu = cv2.threshold(reverse, 180, 255, cv2.THRESH_BINARY)

    # do erosion and dilate
    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_otsu, kernel, iterations=1)
    image_dilate = cv2.dilate(image_erosion, kernel, iterations=1)

    # contour detection
    contours, hierarchy = cv2.findContours(image_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print('length of the first contour: ', int( cv2.arcLength(contours[0], True)))
    print('total number of contour', len(contours))

    # show contour
    draw_img = image.copy()
    res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 1)

    # Draw boundary rectangle
    cnt = contours[0]   # Get the coordinates of the contour
    x, y, w, h = cv2.boundingRect(cnt)
    rec_image = image.copy()
    img = cv2.rectangle(rec_image, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)
    print('coordinate is ', (x - 20, y - 20, x + w + 20, y + h + 20))
    center = (x + (w // 2), y + (w // 2))
    print(center)

    # Crop the image to get the key parts of the image
    cat = cv2.resize(binary_otsu[y - 20:y + h + 20, x - 20:x + w + 20], (120, 150))
    cv2.imshow('cat', cat)
    cv2.waitKey()


    # show all images
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('origin image')
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    ax2.set_title('gray image')
    ax2.axis('off')

    ax3.imshow(cv2.cvtColor(binary_otsu, cv2.COLOR_BGR2RGB))
    ax3.set_title('reverse image')
    ax3.axis('off')

    ax4.imshow(cv2.cvtColor(image_dilate, cv2.COLOR_BGR2RGB))
    ax4.set_title('after E-D')
    ax4.axis('off')

    ax5.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    ax5.set_title('show contour')
    ax5.axis('off')

    ax6.imshow(cv2.cvtColor(rec_image, cv2.COLOR_BGR2RGB))
    ax6.set_title('cut target')
    ax6.axis('off')

    plt.tight_layout()
    plt.show()



