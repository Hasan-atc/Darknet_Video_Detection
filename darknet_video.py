# Import libraries
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import json


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, dictionary, frame_id):
    # Colored labels dictionary
    color_dict = {
        'person': [0, 255, 255], 'bicycle': [238, 123, 158], 'car': [24, 245, 217], 'motorbike': [224, 119, 227],
        'aeroplane': [154, 52, 104], 'bus': [179, 50, 247], 'train': [180, 164, 5], 'truck': [82, 42, 106],
        'boat': [201, 25, 52], 'traffic light': [62, 17, 209], 'fire hydrant': [60, 68, 169],
        'stop sign': [199, 113, 167],
        'parking meter': [19, 71, 68], 'bench': [161, 83, 182], 'bird': [75, 6, 145], 'cat': [100, 64, 151],
        'dog': [156, 116, 171], 'horse': [88, 9, 123], 'sheep': [181, 86, 222], 'cow': [116, 238, 87],
        'elephant': [74, 90, 143],
        'bear': [249, 157, 47], 'zebra': [26, 101, 131], 'giraffe': [195, 130, 181], 'backpack': [242, 52, 233],
        'umbrella': [131, 11, 189], 'handbag': [221, 229, 176], 'tie': [193, 56, 44], 'suitcase': [139, 53, 137],
        'frisbee': [102, 208, 40], 'skis': [61, 50, 7], 'snowboard': [65, 82, 186], 'sports ball': [65, 82, 186],
        'kite': [153, 254, 81], 'baseball bat': [233, 80, 195], 'baseball glove': [165, 179, 213],
        'skateboard': [57, 65, 211],
        'surfboard': [98, 255, 164], 'tennis racket': [205, 219, 146], 'bottle': [140, 138, 172],
        'wine glass': [23, 53, 119],
        'cup': [102, 215, 88], 'fork': [198, 204, 245], 'knife': [183, 132, 233], 'spoon': [14, 87, 125],
        'bowl': [221, 43, 104], 'banana': [181, 215, 6], 'apple': [16, 139, 183], 'sandwich': [150, 136, 166],
        'orange': [219, 144, 1],
        'broccoli': [123, 226, 195], 'carrot': [230, 45, 209], 'hot dog': [252, 215, 56], 'pizza': [234, 170, 131],
        'donut': [36, 208, 234], 'cake': [19, 24, 2], 'chair': [115, 184, 234], 'sofa': [125, 238, 12],
        'pottedplant': [57, 226, 76], 'bed': [77, 31, 134], 'diningtable': [208, 202, 204], 'toilet': [208, 202, 204],
        'tvmonitor': [208, 202, 204], 'laptop': [159, 149, 163], 'mouse': [148, 148, 87], 'remote': [171, 107, 183],
        'keyboard': [33, 154, 135], 'cell phone': [206, 209, 108], 'microwave': [206, 209, 108], 'oven': [97, 246, 15],
        'toaster': [147, 140, 184], 'sink': [157, 58, 24], 'refrigerator': [117, 145, 137], 'book': [155, 129, 244],
        'clock': [53, 61, 6], 'vase': [145, 75, 152], 'scissors': [8, 140, 38], 'teddy bear': [37, 61, 220],
        'hair drier': [129, 12, 229], 'toothbrush': [11, 126, 158]
    }

    for label, confidence, bbox in detections:
        x, y, w, h = (bbox[0],
                      bbox[1],
                      bbox[2],
                      bbox[3])
        name_tag = str(label)
        for name_key, color_val in color_dict.items():
            if name_key == name_tag:
                color = color_val
                xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                jsonUpdater(dictionary, label, frame_id, xmin, ymin, xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 1)
                cv2.putText(img,
                            label +
                            " [" + str(round(float(confidence) * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
    return img


def jsonUpdater(dictionary, nesne, frame_id, xmin, ymin, xmax, ymax):
    if nesne in dictionary:
        dictionary[nesne].append({
            "frame_id": frame_id,
            "frame_zaman": None,
            "sinirlayici_kutu": {
                "ust_sol": {
                    "x": xmin,
                    "y": ymin
                },
                "alt_sag": {
                    "x": xmin,
                    "y": ymin
                }
            },
            "sinif": None,
            "inis_durumu": None
        })
    else:
        dictionary.update({
            nesne: [
                {
                    "frame_id": frame_id,
                    "frame_zaman": None,
                    "sinirlayici_kutu": {
                        "ust_sol": {
                            "x": xmin,
                            "y": ymin
                        },
                        "alt_sag": {
                            "x": xmax,
                            "y": ymax
                        }
                    },
                    "sinif": None,
                    "inis_durumu": None
                }
            ]
        })



netMain = None
metaMain = None
altNames = None


def YOLO():
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"  # Path to cfg
    weightPath = "./yolov4.weights"  # Path to weights
    # Path to meta data
    metaPath = "./cfg/coco.data"
    # Checks whether file exists otherwise return ValueError
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    # Checks the metaMain, NetMain and altNames. Loads it in script
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    network, class_names, class_colors = darknet.load_network(
        configPath, metaPath, weightPath, batch_size=1)
    # cap = cv2.VideoCapture(0)                                      # Uncomment to use Webcam
    # Local Stored video detection - Set input video
    cap = cv2.VideoCapture("VİDEO PATH")
    # Returns the width and height of capture video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    # Set out for video writer
    out = cv2.VideoWriter(  # Set the Output path for video writer
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), video_fps,
        (frame_width, frame_height))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    # Create image according darknet for compatibility of network
    darknet_image = darknet.make_image(frame_width, frame_height, 3)
    # Load the input frame and write output frame.
    frame_id = 0
    while True:
        prev_time = time.time()
        # Capture frame and return true if frame present
        ret, frame_read = cap.read()
        # For Assertion Failed Error in OpenCV
        # Check if frame present otherwise he break the while loop
        if not ret:
            break

        # Convert frame into RGB from BGR and resize accordingly
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        # Copy that frame bytes to darknet_image
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # Detection occurs at this line and return detections, for customize we can change the threshold.
        detections = darknet.detect_image(
            network, class_names, darknet_image, thresh=0.25)
        # Call the function cvDrawBoxes() for colored bounding box per class
        image = cvDrawBoxes(detections, frame_resized, dictionary, frame_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instant_fps = 1 / (time.time() - prev_time)
        print("FPS :", round(instant_fps, 2))
        frame_id += 1
        # cv2.imshow('Demo', image)                                    # Display Image window
        cv2.waitKey(3)
        # Write that frame into output video
        out.write(image)
    # For releasing cap and out.
    cap.release()
    out.release()
    print(":::Video Write Completed")
    with open("sample.json", "w") as f:
      json.dump(dictionary, f)


if __name__ == "__main__":
    # Calls the main function YOLO()
    dictionary = {}
    YOLO()
