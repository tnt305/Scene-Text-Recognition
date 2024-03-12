from PIL import Image

from model import *
from utils import *
def predict(img_path, data_transforms, text_det_model, text_reg_model, idx_to_char, device, visualize=True):
    """
    Thực hiện Scene Text Recognition với một ảnh bất kỳ.
    
    Parameters:
        img_path (str): Path đến file ảnh.
        data_transforms (transforms.Compose): Hàm tiền xử lý dữ liệu ảnh.
        text_det_model (YOLO): Model YOLO text detection.
        text_reg_model (CRNN): Model CRNN text recognition.
        idx_to_char (dict): Bảng mapping idx->classname.
        device (str): 'cpu' hoặc 'gpu'.
        visualize (bool): Kích hoạt visualize kết quả STR.
        
    Returns:
        predictions (list): Danh sách các kết quả STR trên ảnh.
    """
    # Thực hiện text detection
    bboxes, classes, names, confs = text_detection(img_path, text_det_model)

    # Load ảnh
    img = Image.open(img_path)
    
    # Khai báo list rỗng để chứa kết quả STR
    predictions = []

    # Duyệt qua từng kết quả detection
    for bbox, cls, conf in zip(bboxes, classes, confs):
        x1, y1, x2, y2 = bbox # Lấy tọa độ bbox
        confidence = conf # Lấy confidence score
        detected_class = cls # Lấy mã ID của class 
        name = names[int(cls)] # Lấy tên class theo mã ID

        # Cắt ảnh theo bbox
        cropped_image = img.crop((x1, y1, x2, y2))
        
        # Thực hiện text recognition trên ảnh đã cắt
        transcribed_text = text_recognition(
            cropped_image,
            data_transforms,
            text_reg_model,
            idx_to_char,
            device
        )

        # Thêm kết quả STR vào list predictions
        predictions.append((bbox, name, confidence, transcribed_text))
        
    # Thực hiện visualize kết quả STR nếu có
    if visualize:
        visualize_detections(img, predictions)
    
    return predictions