import cv2 
import numpy as np

def read_img(file_name):
    img = cv2.imread(file_name)
    return img

def edge_detection(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur)
    return edges

def color_quantisation(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Display images
def rescale(frame, scale=0.75):
            width = int(frame.shape[1] * scale)
            height = int(frame.shape[0] * scale)
            dimensions = (width, height)
            return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

while True:
    print("1:for sketch")
    print("2:For cartoon")
    print("any other key to exit")
    selection=input()
    if selection=="2":
# Read image
        while True:
            img_to_cartoon = input("Enter the path of the image: ")
            try:
                img = read_img(img_to_cartoon)
                if img is None:
                    raise FileNotFoundError
                break
            except FileNotFoundError:
                print("Image does not exist. Please enter a valid path.")

        # Parameters
        line_wdt = 9
        blur_value = 7
        totalColors = 6

        # Cartoonify image
        edge_img = edge_detection(img, line_wdt, blur_value)
        cartoon_img = color_quantisation(img, totalColors)

        # Applying cartoon effect
        blurred = cv2.bilateralFilter(cartoon_img, d=7, sigmaColor=200, sigmaSpace=200)
        cartoon = cv2.bitwise_and(blurred, blurred, mask=edge_img)
        # Display original and cartoon images
        resized_img = rescale(img,0.70)
        cv2.imshow("Original Image", resized_img)
        resize_cartoon = rescale(cartoon,0.70)
        cv2.imshow("Cartoonized Image", resize_cartoon)

        cv2.waitKey(0)

    elif selection=="1":
        while True:
            img_to_cartoon = input("Enter the path of the image: ")
            try:
                img = read_img(img_to_cartoon)
                if img is None:
                    raise FileNotFoundError
                break
            except FileNotFoundError:
                print("Image does not exist. Please enter a valid path.")
        # Parameters
        line_wdt = 9
        blur_value = 7
        edge_img = edge_detection(img, line_wdt, blur_value)
        edgeResize=rescale(edge_img,0.40)
        resized_img = rescale(img,0.40)
        cv2.imshow("Original Image", resized_img)
        cv2.imshow("edge image",edgeResize)
        # cv2.imwrite("cartoon1_tom.jpg",edgeResize)

        cv2.waitKey(0)

    else:
        break
