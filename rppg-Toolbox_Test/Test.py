import os
import torch
import numpy as np
import cv2
import re
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from neural_methods.model.DeepPhys import DeepPhys

# 去掉state_dict中"module."前缀
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict

# 加载模型时去掉"module."前缀
def load_state_dict_correctly(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = remove_module_prefix(state_dict)  # 去掉module前缀

    # 仅加载匹配的权重
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key] = state_dict[key]
            else:
                print(f"Ignoring shape mismatch for {key}: pretrained shape {state_dict[key].shape}, model shape {model_state_dict[key].shape}")
        else:
            print(f"Ignoring key {key} not found in the model.")

    model.load_state_dict(model_state_dict, strict=False)
    print("Model weights loaded (ignoring mismatched layers).")

# 数据集类
class BistuTestDataset(Dataset):
    def __init__(self, data_path, preprocess_config=None, target_size=(72, 72)):
        """
        Args:
            data_path (str): 数据集文件夹路径，文件夹下应包含 .mp4 文件
            preprocess_config (dict): 预处理配置
            target_size (tuple): 目标尺寸 (H, W)，默认为 (72, 72)
        """
        self.data_path = data_path
        self.preprocess_config = preprocess_config
        self.target_size = target_size  # 目标尺寸
        self.video_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".mp4")]
        if not self.video_files:
            raise ValueError(f"No mp4 files found in {data_path}")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        filename = os.path.basename(video_path)

        # 从文件名中提取平均心率信息
        match = re.search(r'HR=(\d+)', filename)
        if match:
            hr = float(match.group(1))
        else:
            hr = 0.0
            print(f"Warning: No HR information found in filename: {filename}")

        frames, fps = self._load_video(video_path)

        # 根据预处理配置进行处理
        if self.preprocess_config is not None:
            target_H, target_W = self.target_size
            if self.preprocess_config.get('CROP_FACE', False):
                frames = self.face_crop_resize(
                    frames,
                    self.preprocess_config.get('DYNAMIC_DETECTION', False),
                    self.preprocess_config.get('DYNAMIC_DETECTION_FREQUENCY', 180),
                    target_W,
                    target_H,
                    self.preprocess_config.get('LARGE_FACE_BOX', False),
                    True,  # 使用人脸检测
                    self.preprocess_config.get('LARGE_BOX_COEF', 1.0)
                )
            else:
                frames = [self.adjust_image_size(frame, target_W, target_H) for frame in frames]

        frames = np.array(frames)  # shape: (T, H, W, C)
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  # 确保通道维度在正确位置

        # 确保输入数据有 6 个通道（前 3 通道为 diff_input，后 3 通道为 raw_input）
        if frames.shape[1] == 3:
            frames = torch.cat([frames, frames], dim=1)  # 复制一份作为 raw_input

        # 归一化输入数据
        frames = (frames - frames.mean()) / frames.std()

        return frames, hr, filename, idx  # 返回 idx 用于排序

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames, fps

    def adjust_image_size(self, frame, target_W, target_H):
        """
        调整图像大小到目标尺寸，进行中心裁剪或填充。
        """
        h, w = frame.shape[:2]
        # 如果图像比目标大，则中心裁剪
        if h > target_H or w > target_W:
            start_h = (h - target_H) // 2 if h > target_H else 0
            start_w = (w - target_W) // 2 if w > target_W else 0
            frame = frame[start_h:start_h + target_H, start_w:start_w + target_W]
        # 如果图像比目标小，则填充
        elif h < target_H or w < target_W:
            pad_h = (target_H - h) // 2 if h < target_H else 0
            pad_w = (target_W - w) // 2 if w < target_W else 0
            frame = cv2.copyMakeBorder(frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        # 最后确保尺寸完全一致
        frame = cv2.resize(frame, (target_W, target_H), interpolation=cv2.INTER_AREA)
        return frame

    def face_detection(self, frame, use_larger_box=False, larger_box_coef=1.0):
        cascade_path = '/ltb_work/rppg-Toolbox_MMPD/dataset/haarcascade_frontalface_default.xml'  # 请替换为实际路径
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        if len(faces) < 1:
            print("ERROR: No Face Detected in frame, using full frame")
            return [0, 0, frame.shape[1], frame.shape[0]]
        face = faces[0]
        if use_larger_box:
            x, y, w, h = face
            x = max(0, int(x - (larger_box_coef - 1.0) / 2 * w))
            y = max(0, int(y - (larger_box_coef - 1.0) / 2 * h))
            w = int(w * larger_box_coef)
            h = int(h * larger_box_coef)
            return [x, y, w, h]
        else:
            return face.tolist()

    def face_crop_resize(self, frames, use_dynamic_detection, detection_freq, width, height,
                         use_larger_box, use_face_detection, larger_box_coef):
        num_frames = len(frames)
        if use_dynamic_detection:
            num_dynamic_det = int(np.ceil(num_frames / detection_freq))
        else:
            num_dynamic_det = 1
        face_regions = []
        for i in range(num_dynamic_det):
            idx = i * detection_freq if i * detection_freq < num_frames else 0
            if use_face_detection:
                face_region = self.face_detection(frames[idx], use_larger_box, larger_box_coef)
            else:
                face_region = [0, 0, frames[idx].shape[1], frames[idx].shape[0]]
            face_regions.append(face_region)
        resized_frames = []
        for i, frame in enumerate(frames):
            if use_dynamic_detection:
                region_idx = i // detection_freq
                region_idx = min(region_idx, len(face_regions) - 1)
            else:
                region_idx = 0
            x, y, w_box, h_box = face_regions[region_idx]
            cropped = frame[y:y+h_box, x:x+w_box]
            # 使用 adjust_image_size 确保裁剪后的图像大小匹配目标尺寸
            resized = self.adjust_image_size(cropped, width, height)
            resized_frames.append(resized)
        return resized_frames

def main():
    preprocess_config = {
        'DATA_TYPE': ['DiffNormalized', 'Standardized'],
        'LABEL_TYPE': 'DiffNormalized',
        'DO_CHUNK': True,
        'CHUNK_LENGTH': 180,
        'DYNAMIC_DETECTION': False,
        'DYNAMIC_DETECTION_FREQUENCY': 180,
        'CROP_FACE': True,
        'LARGE_FACE_BOX': True,
        'LARGE_BOX_COEF': 1.5,
        'H': 72,  # 根据预训练模型调整
        'W': 72   # 根据预训练模型调整
    }
    data_path = "/ltb_work/rppg-Toolbox_MMPD/Datasets/Bistu_Data"  # 请根据实际情况修改为视频文件存放路径
    model_path = "/ltb_work/rppg-Toolbox_MMPD/PreTrainedModels/UBFC_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_LabelTypeDiffNormalized_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180/UBFC_UBFC_PURE_deepphys_Epoch29.pth"  # 请根据实际情况修改为模型路径
    output_folder = "./test_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 设置目标尺寸为预训练模型的输入尺寸
    target_size = (72, 72)  # 根据预训练模型调整

    dataset = BistuTestDataset(data_path, preprocess_config=preprocess_config, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # 初始化模型
    model = DeepPhys(in_channels=3, img_size=72)  # 根据预训练模型的输入尺寸设置 img_size
    load_state_dict_correctly(model, model_path, device)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    predictions = {}
    labels = {}

    with torch.no_grad():
        for video_frames, hr, filename, idx in dataloader:
            video_frames = video_frames.to(device)
            
            # 调整形状为 [batch_size * T, channels, height, width]
            batch_size, T, C, H, W = video_frames.shape
            video_frames = video_frames.view(batch_size * T, C, H, W)
            
            # 确保输入张量是连续的
            video_frames = video_frames.contiguous()
            
            # 检查输入形状
            print(f"Input shape: {video_frames.shape}")
            
            pred = model(video_frames)
            pred = pred.view(batch_size, T, -1)
            
            # 将预测结果和标签保存到字典中
            subj_index = filename[0].split('.')[0]  # 从文件名中提取唯一标识符
            if subj_index not in predictions:
                predictions[subj_index] = {}
                labels[subj_index] = {}
            predictions[subj_index][idx] = pred.cpu().numpy()
            labels[subj_index][idx] = hr.cpu().numpy()

    # 对每个视频（文件夹）遍历整合所有预测和标签，并分别保存
    for subj, pred_dict in predictions.items():
        pred_list = []
        label_list = []
        for idx in sorted(pred_dict.keys()):
            pred_chunk = predictions[subj][idx]
            label_chunk = labels[subj][idx]
            pred_list.append(pred_chunk)
            label_list.append(label_chunk)
        # 拼接各个块构成完整序列
        subject_pred = np.concatenate(pred_list, axis=0)  # shape: [T, 1]
        subject_label = np.concatenate(label_list, axis=0)  # shape: [T, 1]

        # 将 3D 数组展平为 2D 数组
        if subject_pred.ndim == 3:
            subject_pred = subject_pred.reshape(-1, subject_pred.shape[-1])  # shape: [T, 1]
        if subject_label.ndim == 3:
            subject_label = subject_label.reshape(-1, subject_label.shape[-1])  # shape: [T, 1]

        # 保存为 txt 文件，文件名中包含视频（文件夹）标识符
        pred_save_path = os.path.join(output_folder, f"{subj}_pred.txt")
        label_save_path = os.path.join(output_folder, f"{subj}_label.txt")
        np.savetxt(pred_save_path, subject_pred, fmt="%.4f")
        np.savetxt(label_save_path, subject_label, fmt="%.4f")
        print(f"Saved predictions and labels for {subj} to {output_folder}")

if __name__ == "__main__":
    main()
