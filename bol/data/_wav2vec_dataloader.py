import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import glob
import soundfile as sf

def postprocess_features(feats):
    if feats.dim() == 2: feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    with torch.no_grad():
        feats = F.layer_norm(feats, feats.shape)
    return feats

def get_feature(batch_sample):
    return postprocess_features(batch_sample)

def get_padding_mask(batch_sample):
    return torch.BoolTensor(batch_sample[0].size(1)).fill_(False)

def get_batch_encoder_input(batch_samples):
    features = [get_feature(batch_sample) for batch_sample in batch_samples]
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    return features


class Wav2VecDataset(Dataset):
    def __init__(self, audio_path):
        self.audio_paths = glob.glob(audio_path + '/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        features = self._get_feature(self.audio_paths[index])
        return features

    def _get_feature(self, filepath):
        wav, sample_rate = sf.read(filepath)
        wav = torch.from_numpy(wav).float()
        return wav    

class Wav2VecDataLoader:
    def __init__( self, train_batch_size, num_workers, train_data_path ):
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers

        train_data_loader = self.create_data_loaders_from_train_dataset(train_data_path, train_batch_size, num_workers)
        self.train_data_loader = train_data_loader

  
    def create_data_loaders_from_train_dataset(self, train_data_path, train_batch_size, num_workers):
        train_dataset = Wav2VecDataset(train_data_path)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=train_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        collate_fn=get_batch_encoder_input)

        return train_data_loader

    def get_train_data_loader(self):
        return self.train_data_loader
