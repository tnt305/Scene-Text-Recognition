import os 
from recognition.process.preprocessing import split_bounding_boxes, extract_data_from_xml
root_dir = 'datasets/ocr_dataset'
dataset_dir = 'datasets/SceneTrialTrain'
img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(dataset_dir)
split_bounding_boxes(img_paths, img_labels, bboxes, root_dir)

img_paths = []
labels = []

with open(os.path.join(root_dir, 'labels.txt'), 'r') as f:
    for label in f:
        labels.append(label.strip().split('\t')[1])
        img_paths.append(label.strip().split('\t')[0])

letters = [char.split(".")[0].lower() for char in labels]
letters = "".join(letters)
# Lọc kí tự trùng
letters = sorted(list(set(list(letters))))

# Chuyển list kí tự thành string
chars = "".join(letters)
                
# Thêm kí tự "blank" vào bộ vocab
blank_char = '-'
chars += blank_char
# Tính vocab size
vocab_size = len(chars)
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
max_label_len = max([len(label) for label in labels])