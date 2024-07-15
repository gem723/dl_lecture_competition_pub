import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample, butter, filtfilt
from sklearn.preprocessing import StandardScaler

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
        processed_file_path = os.path.join(processed_dir, f"{split}_X_processed.pt")
        subject_idxs_path = os.path.join(processed_dir, f"{split}_subject_idxs_processed.pt")
        
        if os.path.exists(processed_file_path) and os.path.exists(subject_idxs_path):
            # 保存済みデータを読み込む
            self.X = torch.load(processed_file_path).to(self.dtype)
            self.subject_idxs = torch.load(subject_idxs_path)
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
            torch.save(self.X, processed_file_path)
            torch.save(self.subject_idxs, subject_idxs_path)
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
