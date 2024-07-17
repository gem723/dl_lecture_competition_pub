import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torchvision import transforms

class CombinedCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths_file, brainwave_dataset, transform=None):
        self.clip_dataset = CLIPDataset(image_paths_file, brainwave_dataset.X, transform)
        self.brainwave_dataset = brainwave_dataset

    def __len__(self):
        return len(self.clip_dataset)

    def __getitem__(self, idx):
        image, _ = self.clip_dataset[idx]
        brainwave, label, subject_idx = self.brainwave_dataset[idx]
        return image, brainwave, label, subject_idx

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths_file, brainwave_data, transform=None):
        self.image_paths = []
        with open(image_paths_file, 'r') as f:
            self.image_paths = [line.strip() for line in f]
        
        self.brainwave_data = brainwave_data
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def fix_image_path(self, image_name):
        if '/' not in image_name:
            # 最後の '_' で分割してクラス名とファイル名を分離
            parts = image_name.rsplit('_', 1)
            if len(parts) == 2:
                class_name, file_name = parts
                image_name = f"{class_name}/{image_name}"
            else:
                print(f"Warning: Unexpected image name format: {image_name}")
        return image_name

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        
        # パスの修正処理を__getitem__内で行う
        image_name = self.fix_image_path(image_name)

        class_name, file_name = image_name.split('/')
        image_path = os.path.join('data', 'images', 'Images', class_name, file_name)
        
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            # エラーが発生した場合、最初に見つかった有効な画像を使用
            for alt_image_name in self.image_paths:
                alt_image_name = self.fix_image_path(alt_image_name)
                
                alt_class_name, alt_file_name = alt_image_name.split('/')
                alt_image_path = os.path.join('data', 'images', 'Images', alt_class_name, alt_file_name)
                if os.path.exists(alt_image_path):
                    image_path = alt_image_path
                    print(f"Using alternative image: {image_path}")
                    break
            else:
                raise FileNotFoundError(f"No valid image found in the dataset")

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        brainwave = self.brainwave_data[idx]
        return image, brainwave

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", dtype: torch.dtype = torch.float32, processed_dir: str = "processed", original_rate: int = 200, target_rate: int = 100, lowcut: float = 12.0, highcut: float = 30.0) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.dtype = dtype
        self.processed_dir = processed_dir
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.lowcut = lowcut
        self.highcut = highcut
        
        # 保存済みデータのパス
        processed_file_path = os.path.join(processed_dir, f"{split}_processed.pt")

        if os.path.exists(processed_file_path):
            # 保存済みデータを読み込む
            data_dict = torch.load(processed_file_path)
            self.X = data_dict['X'].to(self.dtype)
            self.subject_idxs = data_dict['subject_idxs']
            if 'y' in data_dict:
                self.y = data_dict['y']
            print(f"Loaded processed data from {processed_file_path}")
        else:
            # 元データを読み込んで前処理を行う
            self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).to(self.dtype)
            self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

            if split in ["train", "val"]:
                self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
                assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

            # データを前処理
            self.preprocess()

            # 前処理後のデータを保存
            os.makedirs(processed_dir, exist_ok=True)
            data_dict = {
                'X': self.X,
                'subject_idxs': self.subject_idxs
            }
            if hasattr(self, 'y'):
                data_dict['y'] = self.y
            torch.save(data_dict, processed_file_path)
            print(f"Saved processed data to {processed_file_path}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def preprocess(self) -> None:
        """データの前処理を行うメソッド"""
        # データをnumpy配列に変換
        data_np = self.X.numpy().astype(self._torch_dtype_to_numpy_dtype(self.dtype))
        # メモリを解放
        del self.X
        torch.cuda.empty_cache()

        # # リサンプリングを適用
        # data_np = self.resample_data(data_np)

        # # フィルタリングを適用
        # data_np = self.bandpass_filter(data_np)
        
        # データをスケーリング
        data_np = self.scale_data(data_np)
        
        # スケーリングされたデータをテンソルに変換
        self.X = torch.tensor(data_np, dtype=self.dtype)
        # メモリを解放
        del data_np
        torch.cuda.empty_cache()
    
    def resample_data(self, data: np.ndarray, chunk_size: int = 10000) -> np.ndarray:
        """データのリサンプリングを行うメソッド（チャンク処理）"""
        print("リサンプリング開始")
        num_trials, num_channels, num_samples = data.shape
        new_num_samples = int(num_samples * self.target_rate / self.original_rate)

        # 結果を格納するための配列を作成
        data_resampled = np.empty((num_trials, num_channels, new_num_samples), dtype=data.dtype)

        for start in range(0, num_trials, chunk_size):
            end = min(start + chunk_size, num_trials)
            data_chunk = data[start:end, :, :]

            # 各チャンクに対してリサンプリングを適用
            for i in range(data_chunk.shape[0]):
                for channel in range(num_channels):
                    data_resampled[start + i, channel, :] = resample(data_chunk[i, channel, :], new_num_samples)

            # メモリの解放
            del data_chunk
            torch.cuda.empty_cache()

        return data_resampled

    def bandpass_filter(self, data: np.ndarray, chunk_size: int = 10000) -> np.ndarray:
        """データにバンドパスフィルタを適用するメソッド（チャンク処理とin-place処理）"""
        print("フィルタリング開始")
        nyquist = 0.5 * self.target_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        # デバッグ用プリント文
        print(f"低域カットオフ: {low}, 高域カットオフ: {high}")

        # 周波数範囲が適切か確認
        if not (0 < low < 1 and 0 < high < 1):
            raise ValueError(f"フィルタのクリティカル周波数が不適切です: low={low}, high={high}")

        b, a = butter(5, [low, high], btype='band')

        num_trials, num_channels, seq_len = data.shape

        for start in range(0, num_trials, chunk_size):
            end = min(start + chunk_size, num_trials)

            # 各チャンクに対してフィルタリングを適用
            for channel in range(num_channels):
                data[start:end, channel, :] = filtfilt(b, a, data[start:end, channel, :], axis=-1)

        return data

    def scale_data(self, data: np.ndarray) -> np.ndarray:
        """データをスケーリングするメソッド（in-place処理）"""
        print("スケーリング開始")
        num_trials, num_channels, seq_len = data.shape

        # 各チャンネルごとにスケーリングを適用
        for channel in range(num_channels):
            channel_data = data[:, channel, :].reshape(-1, 1)
            scaler = StandardScaler()
            scaled_channel_data = scaler.fit_transform(channel_data).reshape(num_trials, seq_len)
            data[:, channel, :] = scaled_channel_data

            # デバッグ用プリント文
            if channel % 10 == 0:  # 10チャンネルごとに出力
                print(f"Processed channel {channel + 1}/{num_channels}")

        return data

    def _torch_dtype_to_numpy_dtype(self, dtype: torch.dtype) -> np.dtype:
        """torch.dtype を numpy.dtype に変換するヘルパーメソッド"""
        if dtype == torch.float32:
            return np.float32
        elif dtype == torch.float16:
            return np.float16
        elif dtype == torch.float64:
            return np.float64
        elif dtype == torch.int32:
            return np.int32
        elif dtype == torch.int64:
            return np.int64
        else:
            raise ValueError(f"Unsupported torch dtype: {dtype}")
