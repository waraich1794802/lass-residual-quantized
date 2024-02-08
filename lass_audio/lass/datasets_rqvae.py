# Path utilities
import librosa
from pathlib import Path
# Pytorch
import torchaudio
from torch.utils.data import Dataset
# Jukebox dataset
from jukebox.data.files_dataset import FilesAudioDataset
from jukebox.hparams import Hyperparams
from jukebox.utils.io import get_duration_sec
import numpy as np

class TrainDataset(FilesAudioDataset):
    def __init__(self, train_dir: str):
        parameters = Hyperparams({
            'sr': 24000,
            'channels': 2,
            'min_duration': 13.65333 + 0.01,
            'max_duration': None,
            'sample_length': int(13.65333 * 24000),
            'aug_shift': True,
            'labels': False,
            'audio_files_dir': train_dir,
        })
        super().__init__(parameters)

    def filter(self, files, durations):
        # Remove files too short or too long
        keep = []
        for i in range(len(files)):
            if durations[i] / self.sr < self.min_duration:
                continue
            if durations[i] / self.sr >= self.max_duration:
                continue
            keep.append(i)
        print(f'self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}');
        print(f"Keeping {len(keep)} of {len(files)} files");
        self.files = [files[i] for i in keep]
        self.durations = [int(durations[i]) for i in keep]
        self.cumsum = np.cumsum(self.durations)

    def init_dataset(self, hps):
        # Load list of files and starts/durations
        files = librosa.util.find_files(f'{hps.audio_files_dir}', ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        print(f"Found {len(files)} files. Getting durations");
        durations = np.array([get_duration_sec(file, cache=True) * self.sr for file in files])  # Could be approximate
        self.filter(files, durations)

'''class TrainDataset(Dataset):
    def __init__(self, directory: str):
        super().__init__()

        self.directory = Path(directory)

        if not self.directory.exists():
            raise ValueError(f"Directory {directory} doesn't exists.");

        self.files = librosa.util.find_files(directory, ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        print(f"Found {len(self.files)} files in {self.directory}");

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.directory / self.files[idx]

        x, _ = torchaudio.load(str(path))
        x = x[None, :, :22000 * 10]

        return { 'audio': x }'''

class MixtureDataset(Dataset):
    def __init__(self, directory_1: str, directory_2: str):
        super().__init__()

        self.directory_1 = Path(directory_1)
        self.directory_2 = Path(directory_2)
        
        if not self.directory_1.exists():
            raise ValueError(f"Directory {directory_1} doesn't exists.");
        if not self.directory_2.exists():
            raise ValueError(f"Directory {directory_2} doesn't exists.");

        self.files_1 = librosa.util.find_files(directory_1, ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        self.files_2 = librosa.util.find_files(directory_2, ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        
        print(f"Found {len(self.files_1)} files in first directory: {self.directory_1}");
        print(f"Found {len(self.files_2)} files in second directory: {self.directory_2}");

    def __len__(self):
        # Returns the shortest length
        return len(self.files_1) if len(self.files_1) < len(self.files_2) else len(self.files_2)

    def __getitem__(self, idx):
        path_1 = self.directory_1 / self.files_1[idx]
        path_2 = self.directory_2 / self.files_2[idx]

        # TODO: add preprocessing to make the sample rate correct
        source_1, _ = torchaudio.load(str(path_1))
        source_1 = source_1[None, :, :22000 * 10]

        source_2, _ = torchaudio.load(str(path_2))
        source_2 = source_2[None, :, :22000 * 10]

        # Compute mixture from sources
        mixture = 0.5 * source_1 + 0.5 * source_2

        return { 'mixture': mixture }
