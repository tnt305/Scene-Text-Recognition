import ultralytics
import os
import shutil
import yaml
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# data extraction 
def extract_data_from_xml(root_dir):
    """
    Trích xuất dữ liệu từ file words.xml trong bộ IC03
    
    Hàm này dùng để trích các thông tin từ file .xml bao gồm: 
    image paths, image sizes, image labels và bboxes 
    
    Parameters:
        root_dir (str): Đường dẫn đến thư mục root của dataset
    
    Returns:
        tuple: Chứa 4 lists lần lượt là: image paths, image sizes, image labels, và bboxes.
    """
    
    xml_path = os.path.join(root_dir, 'words.xml')
    # Parse file xml
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []

    for img in root:
        bbs_of_img = []
        labels_of_img = []

        for bbs in img.findall('taggedRectangles'):
            for bb in bbs:
                # pass if bbox is not alphabet type
                if not bb[0].text.isalnum():
                    continue
                    
                # pass if bb exist special characters
                if 'é' in bb[0].text.lower() or 'ñ' in bb[0].text.lower():
                    continue
                # Format bbox: (xmin, ymin, bbox_width, bbox_height)
                bbs_of_img.append(
                    [
                        float(bb.attrib['x']), 
                        float(bb.attrib['y']), 
                        float(bb.attrib['width']), 
                        float(bb.attrib['height'])
                    ]
                )
                # add labels into list labels_of_img (reformated into lowercase)
                labels_of_img.append(bb[0].text.lower())

        img_path = os.path.join(root_dir, img[0].text)
        img_paths.append(img_path)

        img_sizes.append((int(img[1].attrib['x']), int(img[1].attrib['y'])))

        bboxes.append(bbs_of_img)

        img_labels.append(labels_of_img)

    return img_paths, img_sizes, img_labels, bboxes

# converter
def convert_to_yolov8_format(image_paths, image_sizes, bounding_boxes):
    """
    Thực hiện normalize bounding box
    
    Parameters:
        image_paths (list): Danh sách chứa các path ảnh.
        image_sizes (list): Danh sách chứa độ phân giải ảnh.
        bounding_boxes (list): Danh sách chứa danh sách bounding box.
    
    Returns:
        yolov8_data (list): Danh sách gồm (image_path, image_size, bboxes)
    """
    yolov8_data = []

    for image_path, image_size, bboxes in zip(image_paths, image_sizes, bounding_boxes):
        image_width, image_height = image_size

        yolov8_labels = []

        for bbox in bboxes:
            x, y, w, h = bbox

            # normalization 
            # current format(x_min, y_min, width, height)
            # yolo format (x_center, y_center, width, height)
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            normalized_width = w / image_width
            normalized_height = h / image_height

            # class by default = 0 
            class_id = 0

            # reformat format label
            # Format: "class_id x_center y_center width height"
            yolov8_label = f"{class_id} {center_x} {center_y} {normalized_width} {normalized_height}"
            yolov8_labels.append(yolov8_label)

        yolov8_data.append((image_path, yolov8_labels))

    return yolov8_data