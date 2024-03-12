import torch
import matplotlib.pyplot as plt

def decode(encoded_sequences, idx_to_char, blank_char='-'):
    """
    Decode encoded label thành string
    
    Parameters:
        encoded_sequences (list): Danh sách các tensor label.
        idx_to_char (dict): Bảng mapping ID -> classname.
        blank_char (str): Kí tự "blank".
        
    Returns:
        decoded_sequences (list): danh sách các label đã được decode.
    """
    # Khai báo list rỗng chứa kết quả decode 
    decoded_sequences = []

    # Duyệt qua từng encoded label 
    for seq in encoded_sequences:
        # Khai báo list rỗng chứa từng kí tự đã decode
        decoded_label = []
        # Duyệt qua từng token 
        for idx, token in enumerate(seq):
            # Bỏ qua token padding (ID=0)
            if token != 0:
                # Lấy kí tự của token đang xét trong idx_to_char
                char = idx_to_char[token.item()]
                # Bỏ qua kí tự "blank"
                if char != blank_char:
                    decoded_label.append(char)
        # Thêm chuỗi đã decode vào list decoded_sequences
        decoded_sequences.append(''.join(decoded_label))

    return decoded_sequences
def text_detection(img_path, text_det_model):
    """
    Xác định vị trí (bbox) các text có trong ảnh.
    
    Parameters:
        img_path (str): Path đến file ảnh.
        text_det_model (YOLO): Model YOLO text detection.
        
    Returns:
        tuple: Bao gồm các thành phần đã xác định được (bboxes, classes, names, confs)
    """
    # Thực hiện detection theo YOLO
    text_det_results = text_det_model(img_path, verbose=False)[0]
    
    # Lấy thông tin bboxes (format xyxy)
    bboxes = text_det_results.boxes.xyxy.tolist()
    # Lấy tên classes, confidence scores
    classes = text_det_results.boxes.cls.tolist()
    names = text_det_results.names
    confs = text_det_results.boxes.conf.tolist()
    
    return bboxes, classes, names, confs


def text_recognition(img, data_transforms, text_reg_model, idx_to_char, device):
    """
    Nhận diện văn bản trong ảnh.
    
    Parameters:
        img (PIL.Image): Object ảnh.
        data_transforms (transforms.Compose): Hàm tiền xử lý ảnh.
        text_reg_model (CRNN): Model CRNN text recognition.
        idx_to_char (dict): Bảng mapping ID->classname.
        device (str): 'cpu' hoặc 'gpu'.
        
    Returns:
        text (str): Văn bản nhận diện được.
    """
    transformed_image = data_transforms(img)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    text_reg_model.eval()
    with torch.no_grad():
        logits = text_reg_model(transformed_image).detach().cpu()
    text = decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)
    
    return text

def visualize_detections(img, detections):
    """
    Visualize kết quả Scene Text Recognition (STR).
    
    Parameters:
        img (PIL.Image): Object ảnh.
        detections (list): Danh sách kết quả STR trên ảnh.
    """
    # Cài đặt khung hình
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')

    # Duyệt qua từng (bbox, classname, conf, text)
    for bbox, detected_class, confidence, transcribed_text in detections:
        x1, y1, x2, y2 = bbox
        # Vẽ bbox và text đã nhận diện
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        plt.text(
            x1, y1 - 10, f"{detected_class} ({confidence:.2f}): {transcribed_text}", 
            fontsize=9, bbox=dict(facecolor='red', alpha=0.5)
        )

    plt.show()