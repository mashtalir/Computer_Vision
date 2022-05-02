import numpy as np
import cv2


def laplacian(image, kernel):
    m, n = image.shape
    laplacian = np.zeros((m, n))

    for i in range(2, m - 1):
        for j in range(2, n - 1):
            d1 = (kernel[0, 0] * image[i - 1, j - 1] + kernel[0, 1] * image[i - 1, j] + kernel[0, 2] * image[
                i - 1, j + 1] +
                           kernel[1, 0] * image[i, j - 1] + kernel[1, 1] * image[i, j] + kernel[1, 2] * image[
                               i, j + 1] +
                           kernel[2, 0] * image[i + 1, j - 1] + kernel[2, 1] * image[i + 1, j] + kernel[2, 2] * image[
                               i + 1, j + 1])
            laplacian[i, j] = d1

    for i in range(m):
        for j in range(n):
            if laplacian[i, j] > 40:
                laplacian[i, j] = 255
            else:
                laplacian[i, j] = 0

    return laplacian


# Kirsch edge detection operator
def kirsch(image):
    m, n = image.shape
    list = []
    kirsch = np.zeros((m, n))
    for i in range(2, m - 1):
        for j in range(2, n - 1):
            d1 = np.square(5 * image[i - 1, j - 1] + 5 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                           3 * image[i, j - 1] - 3 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                           3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d2 = np.square((-3) * image[i - 1, j - 1] + 5 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                           3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                           3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d3 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                           3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                           3 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d4 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] -
                           3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] +
                           5 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d5 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] - 3
                           * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] +
                           5 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d6 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                           5 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] +
                           5 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d7 = np.square(5 * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                           5 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] -
                           3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d8 = np.square(5 * image[i - 1, j - 1] + 5 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                           5 * image[i, j - 1] - 3 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                           3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            # Take the maximum value in each direction, the effect is not good, use another method
            list = [d1, d2, d3, d4, d5, d6, d7, d8]
            kirsch[i, j] = int(np.sqrt(max(list)))
            # : Rounding the die length in all directions
            # kirsch[i, j] = int(np.sqrt(d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8))
    for i in range(m):
        for j in range(n):
            if kirsch[i, j] > 127:
                kirsch[i, j] = 255
            else:
                kirsch[i, j] = 0
    return kirsch


def main():
    image1 = cv2.imread('4.png', cv2.IMREAD_GRAYSCALE)
    krnl_lp_1 = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]])
    krnl_lp_2 = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]])
    output = laplacian(image1, kernel=krnl_lp_1)
    output_4 = cv2.filter2D(image1, -1, krnl_lp_1)
    cv2.imshow('Original image', image1)
    cv2.imshow('image1', output)
    cv2.imshow('image2', output_4)
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
