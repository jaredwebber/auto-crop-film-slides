import cv2
from cv2.typing import MatLike
import time
import os
import sys
from numpy import ndarray

CROP_ITERATIONS: int = 3
CROPPING_THRESHOLD: float = 10  # 10 recommended for film slides


def process_files(dir_path: str) -> None:
    for i in range(CROP_ITERATIONS):
        print(f"Starting Pass {i+1}...")
        for filename in os.scandir(dir_path):
            if filename.is_file():
                image: MatLike
                try:
                    image = cv2.imread(filename.path)
                    if image is None:
                        raise Exception("Failed to load the image.")
                except Exception as e:
                    print(f"Failed to read image: {filename.path} - {e}")
                    continue

                start_time: float = time.perf_counter()
                crop_arr: ndarray = remove_border(image)
                print(
                    f"{filename.name} - processed in {(time.perf_counter() - start_time):.3f} seconds"
                )
                cv2.imwrite(filename.path, crop_arr)
        print("\n\n")


def remove_border(image) -> ndarray:
    # Convert to grayscale
    gray: MatLike = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, CROPPING_THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour: MatLike = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y : y + h, x : x + w]  # noqa: E203

    return cropped_image


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Invalid Parameters, make sure directory path is included")
    dir_path: str = sys.argv[1]
    process_files(dir_path)
