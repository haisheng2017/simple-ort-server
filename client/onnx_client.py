import http.client
import json
import cv2
import base64
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.utils.detection_utils import DetectionVisualization
from super_gradients.training.utils.media.image import load_image


def show_predictions_from_batch_format(image, predictions, original_shape):
    num_predictions = predictions["graph2_num_predictions"]
    pred_boxes = predictions["graph2_pred_boxes"]
    pred_scores = predictions["graph2_pred_scores"]
    pred_classes = predictions["graph2_pred_classes"]
    assert (
        len(num_predictions) == 1
    ), "Only batch size of 1 is supported by this function"
    num_predictions = num_predictions[0][0]
    pred_boxes = pred_boxes[0]
    pred_scores = pred_scores[0]
    pred_classes = pred_classes[0]
    image = image.copy()
    class_names = COCO_DETECTION_CLASSES_LIST
    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

    for it in range(num_predictions):
        x1 = pred_boxes[it][0]
        y1 = pred_boxes[it][1]
        x2 = pred_boxes[it][2]
        y2 = pred_boxes[it][3]
        class_score = pred_scores[it]
        class_index = pred_classes[it]
        image = DetectionVisualization.draw_box_title(
            image_np=image,
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            class_id=class_index,
            class_names=class_names,
            color_mapping=color_mapping,
            box_thickness=2,
            pred_conf=class_score,
        )
    image = cv2.resize(
        image,
        dsize=(original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.tight_layout()
    plt.show()


def infer(host: str, modelPath: str, imagePath: str):
    cwd = os.getcwd()

    headers = {"Content-type": "application/json", "Accept": "application/json"}
    name = "test"
    body = {"modelFile": os.path.join(cwd, modelPath), "name": name}
    conn = http.client.HTTPConnection(host)
    conn.request("POST", "/onnx", body=json.dumps(body), headers=headers)
    response = conn.getresponse()
    print(response.status, response.reason)
    data = response.read()
    name_set = json.loads(data)
    print(name_set)
    conn.request("GET", "/onnx/meta?name=" + name, headers=headers)
    print(response.status, response.reason)
    response = conn.getresponse()
    data = response.read()
    metadata = json.loads(data)
    shape = metadata["inputMeta"]["input"]["info"]["shape"]
    print(shape)
    img = load_image(imagePath)
    original_shape = img.shape
    print(original_shape)
    img = cv2.resize(img, dsize=(shape[3], shape[2]), interpolation=cv2.INTER_LINEAR)
    image_bchw = np.transpose(np.expand_dims(img, 0), (0, 3, 1, 2))
    print(image_bchw.shape)
    image_b64 = base64.b64encode(image_bchw.tobytes()).decode()
    # print(image_b64)
    body = {"name": name, "imageB64": image_b64, "inputName": "input"}
    conn.request("POST", "/onnx/infer", body=json.dumps(body), headers=headers)
    response = conn.getresponse()
    print(response.status, response.reason)
    data = response.read()
    result = json.loads(data)
    # print(result)
    show_predictions_from_batch_format(img, result, original_shape)
    conn.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="示例客户端")
    parser.add_argument(
        "-m",
        dest="modelPath",
        required=True,
        help="模型文件本地地址",
    )
    parser.add_argument(
        "-f",
        dest="imageFile",
        required=True,
        help="Jpeg图片本地地址",
    )
    parser.add_argument("--host", dest="host", required=True, help="服务器地址")
    args = parser.parse_args()
    print(args)
    infer(
        args.host,
        args.modelPath,
        args.imageFile,
    )
