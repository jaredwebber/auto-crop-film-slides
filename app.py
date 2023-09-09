import cv2
import time
import os
import sys
from numpy import ndarray, array
from PIL import Image as ImageModule
from PIL.Image import Image


def process_files(dir_path: str) -> None:
    for filename in os.scandir(dir_path):
        portrait: bool = False
        if filename.is_file():
            file: Image
            try:
                file = ImageModule.open(filename.path).convert("RGB")
                h, w, c = cv2.imread(filename.path).shape
                portrait = h > w
            except Exception as e:
                print(f"Failed to open Image: {e}")
                continue

            start_time: float = time.perf_counter()
            crop_arr: ndarray = remove_border(file, portrait)
            print(f"Process time: {(time.perf_counter() - start_time):.3f} seconds")

            filepath: str = ""
            file_extension: str = ""
            try:
                filepath, file_extension = filename.path.split(".")
            except Exception as e:
                print(f"Failed to parse filepath: {filepath}, {e}")

            ImageModule.fromarray(crop_arr).save(f"{filepath}_cropped.{file_extension}")


# Based on https://github.com/hafidh561/detect-and-remove-black-border-from-screenshots-images
def remove_border(image: Image, portrait: bool) -> ndarray:
    image_arr: ndarray = array(image)
    image_gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = array([])
    max_area = 0
    for cntrs in contours:
        area: float = cv2.contourArea(cntrs)
        peri: float = cv2.arcLength(cntrs, True)
        approx = cv2.approxPolyDP(cntrs, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    cnt = biggest
    x: int
    y: int
    if not portrait:
        x, y, w, h = cv2.boundingRect(cnt)
    else:
        y, x, w, h = cv2.boundingRect(cnt)
    crop = image_arr[y : y + h, x : x + w]  # noqa: E203
    return crop


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Invalid Parameters, make sure directory path is included")
    dir_path: str = sys.argv[1]
    process_files(dir_path)
