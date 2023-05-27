import cv2
import os
import filetype
import numpy as np


AGE_MODEL = "weights/deploy_age.prototxt"
AGE_PROTO = "weights/age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

FACE_PROTO = "weights/deploy.prototxt.txt"

FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"


frame_width = 1280
frame_height = 720


face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)


def get_faces(frame, confidence_threshold=0.5):

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    
    face_net.setInput(blob)
    
    output = np.squeeze(face_net.forward())
    
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            
            start_x, start_y, end_x, end_y = box.astype(np.int64)
            start_x, start_y, end_x, end_y = start_x - \
                                             10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale / 10
    return 1

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def predict_age():

    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        frame = img.copy()
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        faces = get_faces(frame)
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            blob = cv2.dnn.blobFromImage(
                image=face_img, scalefactor=1.0, size=(227, 227),
                mean=MODEL_MEAN_VALUES, swapRB=False
            )
            age_net.setInput(blob)
            age_preds = age_net.forward()
            print("=" * 30, f"Face {i + 1} Prediction Probabilities", "=" * 30)
            for i in range(age_preds[0].shape[0]):
                print(f"{AGE_INTERVALS[i]}: {age_preds[0, i] * 100:.2f}%")
            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            age_confidence_score = age_preds[0][i]
            label = f"Age:{age} - {age_confidence_score * 100:.2f}%"
            print(label)
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            cv2.putText(frame, label, (start_x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
       
        cv2.imshow('Age Estimator', frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_age()
