import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def img2label(img):
    img = img[:, : img.shape[1] // 2]
    x_min = int(img.shape[0] * 0.2)
    x_max = img.shape[0] - x_min

    y_min = int(img.shape[1] * 0.1)
    y_max = img.shape[1] - y_min
    img = img[x_min: x_max, y_min: y_max]
    ret, binary = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    # print(binary.shape)
    binary = cv2.cvtColor(binary, cv2.COLOR_RGB2GRAY)
    im_floodfill = binary.copy()
    h, w = binary.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = binary | im_floodfill_inv
    return img, im_out


if __name__ == '__main__':
    img = cv2.imread('/media/ubuntu/black/zhirui/TransUnet/标记图片/1类-无结节/2.JPG')
    img, im_out = img2label(img)
    print(img.shape)
    print(im_out.shape)
    plt.figure(figsize=(12, 12))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(im_out, 'gray')
    plt.show()
