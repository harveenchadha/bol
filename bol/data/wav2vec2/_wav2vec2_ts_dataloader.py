import torch
import torchaudio
from torch.utils.data import Dataset
from bol.utils.helper_functions import read_wav_file, convert_audio


def get_batch_encoder_input(batch_samples):
    # features = [get_feature(batch_sample[0]) for batch_sample in batch_samples]
    features = [batch_sample[0].squeeze(dim=0) for batch_sample in batch_samples]

    filenames = [filename[1] for filename in batch_samples]
    # features = batch_samples[0][0]
    # filenames = batch_samples[1][0]
    # print(features)
    # print(features[0])
    # print("Zer: ", features[0].size())

    # print("Max size is :", max(sizes))

    #    for feature in features:

    # features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    return features, filenames


class Wav2Vec2TsDataSet(Dataset):
    def __init__(self, audio_path):
        self.audio_paths = audio_path

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        features = self._get_feature(self.audio_paths[index])
        return features, self.audio_paths[index]

    def _get_feature(self, filepath):
        wav, sample_rate = read_wav_file(filepath, 'ta')
        if sample_rate != 16000:
            # wav = convert_audio(wav, sample_rate, 16000, 'ta')
            if sample_rate!= 16000: #hardcoding
                wav = convert_audio( filepath, sample_rate, 16000, 'sox')
                new_path = "/tmp/" + filepath.split('/')[-1]
                wav, sample_rate = read_wav_file(new_path, 'ta')
        return wav


class Wav2Vec2TsDataLoader:
    def __init__(self, batch_size, num_workers, file_data_path):
        self.batch_size = batch_size
        self.num_workers = num_workers

        file_data_loader = self.create_data_loaders_from_dataset(
            file_data_path, batch_size, num_workers
        )
        self.file_data_loader = file_data_loader

    def create_data_loaders_from_dataset(self, file_data_path, batch_size, num_workers):
        train_dataset = Wav2Vec2TsDataSet(file_data_path)
        file_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        # collate_fn=get_batch_encoder_input)

        return file_data_loader

    def get_file_data_loader(self):
        return self.file_data_loader
