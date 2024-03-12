import torch
import os

def encode(label, char_to_idx, max_label_len):
    """
    Encode label thành tensor
    
    Paramaters:
        label (str): String label. 
        char_to_idx (dict): Bảng mapping classname -> ID.
        max_label_len (int): Độ dài tối đa cho label.
        
    Returns:
        padded_labels (tensor): Tensor label đã được encode và padding.
        lengths (tensor): Độ dài trước khi padding của label.
    """
    # Đổi sang tensor 
    encoded_labels = torch.tensor(
        [char_to_idx[char] for char in label], 
        dtype=torch.long
    )
    # Tính len của label 
    label_len = len(encoded_labels)
    lengths = torch.tensor(
        label_len, 
        dtype=torch.long
    )
    # Padding 
    padded_labels = F.pad(
        encoded_labels, 
        (0, max_label_len - label_len), 
        value=0
    )
    
    return padded_labels, lengths

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