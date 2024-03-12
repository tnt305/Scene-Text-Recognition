from ultralytics import YOLO
import torch

from recognition.model import CRNN
from recognition.process import vocab

device = 'cuda' if torch.cuda.is_available() else 'cpu'
text_det_model_path = 'models/yolov8/detect/train/weights/best.pt'
text_reg_mode_path = 'models/ocr_crnn_resnet_best.pt'
yolo = YOLO(text_det_model_path)

crnn_model = CRNN(  vocab_size=vocab.vocab_size,
                    hidden_size=256,
                    n_layers=2,
                    dropout=0.3,
                    unfreeze_layers=2
                ).to(device)
crnn_model.load_state_dict(torch.load(text_reg_mode_path))