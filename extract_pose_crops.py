import json
import numpy as np
import os
import sys
import subprocess
import cv2
import glob

ADAPTIVE_SCALE = 1.5  # crop bbox = 150% of pose detection bbox


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (tuple): (x1, y1, x2, y2) for the first box
        box2 (tuple): (x1, y1, x2, y2) for the second box

    Returns:
        float: The IoU score.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area


def extract_frame_ffmpeg(video_path, timestamp_sec, output_path):
    """Extract a single frame from a video at the given timestamp using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [ffmpeg error] {result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.returncode == 0


def visualize_boxes(image_path, boxes, output_path):
    """Draw bounding boxes on the image and save to output_path."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [warn] Could not read image for visualization: {image_path}")
        return
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
        label = f"{x2-x1}x{y2-y1}"
        cv2.putText(img, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    time_interval = 60   # seconds between sampled frames
    fps = 3
    datalen = 1800

    im_w = 2560
    im_h = 1440

    iou_threshold = 0.3

    dataroot = sys.argv[1]
    os.chdir(dataroot)

    files = []
    for f in glob.glob("*.mp4"):
        files.append(f[:-4])

    print(f"Found {len(files)} video(s): {files}")

    crop_file_path = os.path.join(dataroot, "crop.txt")
    with open(crop_file_path, "w", encoding="utf-8", newline="\n") as crop_file_f:
        for file in files:
            video_path = os.path.join(dataroot, file + ".mp4")
            pose_path = os.path.join(dataroot, file + ".json")

            # Load pose annotations indexed by frame id
            annotations = [[] for _ in range(datalen)]
            if os.path.exists(pose_path):
                with open(pose_path) as f:
                    json_annotations = json.load(f)
                for ann in json_annotations:
                    image_id = int(ann["image_id"][:-4])
                    if image_id < datalen:
                        annotations[image_id].append(ann)
            else:
                print(f"  [warn] Pose file not found: {pose_path}")

            for t in range(0, datalen, time_interval):
                if not annotations[t]:
                    continue

                timestamp_sec = t / fps
                image_name = f"{file}_{t:06d}.jpg"
                image_path = os.path.join(dataroot, image_name)

                # --- Extract frame with ffmpeg ---
                print(f"  Extracting frame t={t} ({timestamp_sec:.1f}s) -> {image_name}")
                success = extract_frame_ffmpeg(video_path, timestamp_sec, image_path)
                if not success:
                    print(f"  [skip] Frame extraction failed for t={t}")
                    continue

                # --- Compute adaptive crop boxes ---
                # ann["box"] is expected as [cx, cy, pose_w, pose_h] (center x/y, bbox dims)
                ann_set_x1y1x2y2 = []
                for ann in annotations[t]:
                    cx, cy = ann["box"][0], ann["box"][1]
                    pose_w, pose_h = ann["box"][2], ann["box"][3]

                    crop_w = pose_w * ADAPTIVE_SCALE
                    crop_h = pose_h * ADAPTIVE_SCALE

                    x = cx - 0.5 * crop_w
                    y = cy - 0.5 * crop_h

                    # Clamp to image bounds
                    if x < 0:
                        x = 0
                    elif x + crop_w > im_w:
                        x = im_w - crop_w

                    if y < 0:
                        y = 0
                    elif y + crop_h > im_h:
                        y = im_h - crop_h

                    ann_set_x1y1x2y2.append([x, y, x + crop_w, y + crop_h])

                if not ann_set_x1y1x2y2:
                    continue

                # --- IoU-based NMS ---
                n = len(ann_set_x1y1x2y2)
                iou_matrix = np.zeros((n, n))
                for idx1, b1 in enumerate(ann_set_x1y1x2y2):
                    for idx2, b2 in enumerate(ann_set_x1y1x2y2):
                        iou_matrix[idx1, idx2] = calculate_iou(b1, b2)

                iou_agg = iou_matrix.sum(axis=0) - 1  # exclude self-overlap

                # cv2.dnn.NMSBoxes expects [x, y, w, h]
                nms_boxes = [
                    [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
                    for b in ann_set_x1y1x2y2
                ]
                indices = cv2.dnn.NMSBoxes(
                    bboxes=nms_boxes,
                    scores=iou_agg.tolist(),
                    score_threshold=0.0,
                    nms_threshold=iou_threshold,
                )

                if len(indices) == 0:
                    continue

                filtered_boxes = [ann_set_x1y1x2y2[i] for i in indices.flatten()]

                # --- Write crop.txt entries: image_name.jpg x y w h ---
                for box in filtered_boxes:
                    x1 = int(box[0])
                    y1 = int(box[1])
                    w = int(box[2] - box[0])
                    h = int(box[3] - box[1])
                    crop_file_f.write(f"{image_name} {x1} {y1} {w} {h}\n")

                # --- Visualize bounding boxes ---
                viz_path = os.path.join(dataroot, f"{file}_{t:06d}_viz.jpg")
                visualize_boxes(image_path, filtered_boxes, viz_path)
                print(f"  Saved {len(filtered_boxes)} box(es) -> viz: {os.path.basename(viz_path)}")

    print(f"\nDone. Crop entries written to: {crop_file_path}")
