import torch
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
    features = [get_feature(batch_sample[0]) for batch_sample in batch_samples]
    filenames = [filename[1] for filename in batch_samples]
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    return features, filenames


class Wav2VecDataset(Dataset):
    def __init__(self, audio_path):
        self.audio_paths = glob.glob(audio_path + '/**/*.wav', recursive=True)[0:20]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        features = self._get_feature(self.audio_paths[index])
        return features, self.audio_paths[index]

    def _get_feature(self, filepath):
        wav, sample_rate = sf.read(filepath)
        wav = torch.from_numpy(wav).float()
        #wav = wav.to('cuda')
        return wav    

class Wav2VecDataLoader:
    def __init__( self, train_batch_size, num_workers, file_data_path ):
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers

        file_data_loader = self.create_data_loaders_from_train_dataset(file_data_path, train_batch_size, num_workers)
        self.file_data_loader = file_data_loader

  
    def create_data_loaders_from_train_dataset(self, file_data_path, train_batch_size, num_workers):
        train_dataset = Wav2VecDataset(file_data_path)
        file_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=train_batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        collate_fn=get_batch_encoder_input)

        return file_data_loader

    def get_file_data_loader(self):
        return self.file_data_loader
