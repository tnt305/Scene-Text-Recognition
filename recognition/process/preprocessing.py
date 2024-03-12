import os
import torchvision
from torchvision import transforms
import shutil
import xml.etree.ElementTree as ET
from PIL import Image


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
    
    # Tạo path đến file words.xml 
    xml_path = os.path.join(root_dir, 'words.xml')
    # Parse file xml
    tree = ET.parse(xml_path)
    # Đọc thẻ root của file
    root = tree.getroot()

    # Khai báo các list rỗng để lưu dữ liệu
    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []

    # Duyệt qua từng thẻ ảnh <image>
    for img in root:
        # Khai báo các list rỗng chứa bboxes và labels của ảnh đang xét
        bbs_of_img = []
        labels_of_img = []

        # Duyệt qua từng thẻ boundingbox 
        for bbs in img.findall('taggedRectangles'):
            for bb in bbs:
                # Bỏ qua trường hợp label không phải kí tự alphabet hoặc number
                if not bb[0].text.isalnum():
                    continue
                    
                # Bỏ qua trường hợp label là chữ 'é' hoặc ñ'
                if 'é' in bb[0].text.lower() or 'ñ' in bb[0].text.lower():
                    continue

                # Đưa thông tin tọa độ bbox vào list bbs_of_img
                # Format bbox: (xmin, ymin, bbox_width, bbox_height)
                bbs_of_img.append(
                    [
                        float(bb.attrib['x']), 
                        float(bb.attrib['y']), 
                        float(bb.attrib['width']), 
                        float(bb.attrib['height'])
                    ]
                )
                # Đưa label vào list labels_of_img (đã chuyển chữ viết thường)
                labels_of_img.append(bb[0].text.lower())
                
        # Đưa thông tin path ảnh đang xét vào list img_paths
        img_path = os.path.join(root_dir, img[0].text)
        img_paths.append(img_path)
        # Đưa thông tin độ phân giải ảnh vào list img_sizes
        img_sizes.append((int(img[1].attrib['x']), int(img[1].attrib['y'])))
        # Đưa list bbox vào list bboxes
        bboxes.append(bbs_of_img)
        # Đưa list labels vào list img_labels
        img_labels.append(labels_of_img)

    return img_paths, img_sizes, img_labels, bboxes

def split_bounding_boxes(img_paths, img_labels, bboxes, save_dir):
    """
    Xây dựng thư mục chứa dữ liệu cho Text Recognition.
    
    Hàm sẽ tạo một thư mục save_dir, lưu các ảnh cắt từ tọa độ bbox.
    Label sẽ được lưu riêng vào file labels.txt. 
    
    Parameters:
        img_paths (list): Danh sách các path ảnh.
        img_labels (list): Danh sách chứa danh sách labels của các ảnh.
        bboxes (list): Danh sách chứa danh sách bounding box của các ảnh.
        save_dir (str): Đường dẫn đến thư mục chứa dữ liệu.
    """
    # Tạo tự động thư mục chứa dữ liệu
    os.makedirs(save_dir, exist_ok=True)

    # Khai báo biến đếm và danh sách rỗng chứa labels
    count = 0
    labels = []  

    # Duyệt qua từng cặp (đường dẫn ảnh, list label, list bbox)
    for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
        # Đọc ảnh
        img = Image.open(img_path)

        # Duyệt qua từng cặp label và bbox
        for label, bb in zip(img_label, bbs):
            # Cắt ảnh theo bbox
            cropped_img = img.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))

            # Bỏ qua trường hợp 90% nội dung ảnh cắt là màu trắng hoặc đen.
            if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                continue
                
            # Bỏ qua trường hợp ảnh cắt có width < 10 hoặc heigh < 10
            if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
                continue
                
            # Bỏ qua trường hợp số kí tự của label < 3
            if len(label) < 3:
                continue

            # Tạo tên cho file ảnh đã cắt và lưu vào save_dir
            filename = f"{count:06d}.jpg"
            cropped_img.save(os.path.join(save_dir, filename))

            new_img_path = os.path.join(save_dir, filename)

            # Đưa format label mới vào list labels
            # Format: img_path\tlabel
            label = new_img_path + '\t' + label

            labels.append(label)  # Append label to the list

            count += 1

    print(f"Created {count} images")

    # Đưa list labels vào file labels.txt
    with open(os.path.join(save_dir, 'labels.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")


data_transforms = {
    # Dành cho dữ liệu train 
    'train': transforms.Compose([
        transforms.Resize((32, 100)), 
        transforms.ColorJitter( 
            brightness=0.5, 
            contrast=0.5, 
            saturation=0.5
        ),
        transforms.Grayscale(num_output_channels=1),
        transforms.GaussianBlur(3),
        transforms.RandomAffine(degrees=2, shear=2),  
        transforms.RandomPerspective(
            distortion_scale=0.4, 
            p=0.5, 
            interpolation=3
        ),  
        transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)), 
    ]),
    # Dành cho dữ liệu val, test
    'val': transforms.Compose([
        transforms.Resize((32, 100)), 
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)), 
    ]),
}