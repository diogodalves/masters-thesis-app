import cv2


def init_haar_cascade(model_path) -> None:
    global MODEL
    MODEL = cv2.CascadeClassifier(model_path)


def get_haar_cascade():
    return MODEL