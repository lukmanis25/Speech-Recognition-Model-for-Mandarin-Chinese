import pathlib
from argparse import ArgumentParser

import torch
import torchaudio

from data import PADDING, DataHandler
from model import PinyinTransformer
from torch import nn
from training import N_TOKENS


class Main:
    def __init__(self, model_path: pathlib.Path, n_inputs: int, n_hidden: int, n_layers: int, batch_size: int):
        print("generating index")
        data_handler = DataHandler(
            phone_set_path=pathlib.Path("AISHELL-3/phone-set.txt"),
            train_path=pathlib.Path("AISHELL-3/train/train_content.txt"),
            test_path=pathlib.Path("AISHELL-3/test/test_content.txt")
        )
        test_set = data_handler.get_set("test")

        if self.device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        # create model object
        n_head = 1 # nb of heads for multiheaded attention
        self.model = PinyinTransformer(N_TOKENS, n_inputs, n_head, n_hidden, n_layers).double().to(self.device)

        # load the model
        self.load_model(model_path)

        test_set = data_handler.get_set("test")
        _, sample_rate, _ = test_set[0]
        print("done generating sets")

        self.loss_criterion = nn.CrossEntropyLoss(ignore_index=data_handler.index.phon2idx[PADDING])

        # loading waveform from .wav:
        #   waveform, sample_rate = torchaudio.load("audio_file.wav", normalize=True)
        new_sample_rate = 256 # 8000 # 16000 # original: 44100
        self.transform_resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        self.transform_mfcc = torchaudio.transforms.MFCC(sample_rate=new_sample_rate, n_mfcc=n_inputs, melkwargs={"n_mels": n_inputs}) # (sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 80, "n_mels": 23, "center": False}) # n_mfcc=13 (default:40), , log_mels=True

        self.transform_resample = self.transform_resample.double().to(self.device)
        self.transform_mfcc = self.transform_mfcc.double().to(self.device)

        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=data_handler.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.run()


    def load_model(self, model_path: pathlib.Path):
        checkpoint = torch.load(model_path) # str(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        print("loaded the model successfully")

    def run(self):
        with torch.no_grad():
            # TODO: run model.recognize on test split here
            # TODO: later on try with our data
            pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=lambda p: pathlib.Path(p).absolute(), dest="model_path",
                        help="optional path to previously saved model")
    args = parser.parse_args()
    print(f"path: {args.model_path}")
    Main()