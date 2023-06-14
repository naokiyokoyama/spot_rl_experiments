import cv2
import torch
from transformers import OwlViTForObjectDetection, OwlViTProcessor


class OwlViT:
    def __init__(self, texts, device="cuda"):
        model_name = "google/owlvit-base-patch32"
        # model_name = "google/owlvit-base-patch16"
        # model_name = "google/owlvit-large-patch14"
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.texts = [texts]

    def detect_objects(self, img, score_threshold=0.1):
        # Convert image from BGR to RGB
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            text=self.texts, images=img_in, return_tensors="pt"
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.shape[:2]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes
        )

        return results

    def visualize_results(self, img, results, score_threshold=0.1):
        viz_img = img.copy()

        i = 0  # assume only one set of classes are being used
        text = self.texts[i]
        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )
        for box, score, label in zip(boxes, scores, labels):
            box = [int(i) for i in box.tolist()]
            if score >= score_threshold:
                viz_img = draw_bounding_box(
                    viz_img,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    self.label_idx_to_color(label.item()),
                    text[label],
                    score,
                )
        return viz_img

    def label_idx_to_color(self, label_idx):
        # Uniformly sample colors from the COLORMAP_RAINBOW
        color = cv2.applyColorMap(
            np.uint8([255 * (label_idx + 1) / len(self.texts[0])]),
            cv2.COLORMAP_RAINBOW,
        )[0][0]
        color = tuple(int(i) for i in color)
        return color

    def pred2string(self, results, classes_to_id):
        i = 0  # assume only one set of classes are being used
        text = self.texts[i]
        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )
        detection_str = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [int(i) for i in box.tolist()]
            class_id = classes_to_id[text[label]]
            det_attrs = [
                str(i) for i in [class_id, score.item(), x1, y1, x2, y2]
            ]
            detection_str.append(",".join(det_attrs))
        if len(detection_str) == 0:
            return "None"
        detection_str = ";".join(detection_str)
        return detection_str


def draw_bounding_box(image, point1, point2, color, class_name, score):
    # Create a copy of the input image to draw on
    img = image.copy()

    # Draw bounding box on image
    box_thickness = 2
    cv2.rectangle(img, point1, point2, color, thickness=box_thickness)

    # Draw class name and score on image
    text_label = f"{class_name}: {int(score * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size, _ = cv2.getTextSize(
        text_label, font, font_scale, font_thickness
    )
    text_x = point1[0]
    text_y = point2[1] + text_size[1]
    cv2.rectangle(
        img,
        (text_x, text_y - 2 * text_size[1]),
        (text_x + text_size[0], text_y - text_size[1]),
        color,
        -1,
    )
    cv2.putText(
        img,
        text_label,
        (text_x, text_y - text_size[1] - box_thickness),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness + 1,
    )
    cv2.putText(
        img,
        text_label,
        (text_x, text_y - text_size[1] - box_thickness),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )

    return img


if __name__ == "__main__":
    import urllib

    import numpy as np

    req = urllib.request.urlopen(
        "http://images.cocodataset.org/val2017/000000039769.jpg"
    )
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)

    texts = [
        "a photo of a cat",
        "a photo of a television remote",
        "a photo of a blanket",
        "a photo of a cushion",
    ]

    owl = OwlViT(texts)
    results = owl.detect_objects(image)
    viz_img = owl.visualize_results(image, results)
    cv2.imwrite("output.jpg", viz_img)
