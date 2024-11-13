# import sys
# import numpy
import pathlib
import torch
import torch.nn.functional as F
import torchaudio

from argparse import ArgumentParser
from typing import Callable
from datetime import datetime
from torch import nn, optim, save
from tqdm import tqdm

from data import PADDING, DataHandler
from model import PinyinTransformer
# from torch.utils.tensorboard import SummaryWriter
# from torch import tensor, nn,

# import os
# import random


import torch.nn as nn
m = nn.Transformer()


# pinyin phone set magnitude + controll tokens
N_TOKENS = 3393 + 3

# region def collate_fn(batch):
#     """KOLACJONOWANIE DANYCH"""
#     # A data tuple has the form:
#     # waveform, sample_rate, label, speaker_id, utterance_number

#     tensors, targets = [], []

#     # Gather in lists, and encode labels as indices
#     for waveform, _, label, *_ in batch:
#         tensors += [waveform]
#         targets += [label_to_index(label)]

#     # Group the list of tensors into a batched tensor
#     tensors = pad_sequence(tensors)
#     targets = torch.stack(targets)

#     return tensors, targets
# endregion

# def count_parameters(model):
#     return torch.sum(p.numel() for p in model.parameters() if p.requires_grad)

# def number_of_correct(pred, target):
#     # count number of correct predictions
#     return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

# region class SubsetSC(SPEECHCOMMANDS):
#     def __init__(self, subset: str = None):
#         super().__init__("./", download=True)

#         def load_list(filename):
#             filepath = os.path.join(self._path, filename)
#             with open(filepath) as fileobj:
#                 return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

#         if subset == "validation":
#             self._walker = load_list("validation_list.txt")
#         elif subset == "testing":
#             self._walker = load_list("testing_list.txt")
#         elif subset == "training":
#             excludes = load_list("validation_list.txt") + \
#                 load_list("testing_list.txt")
#             excludes = set(excludes)
#             self._walker = [w for w in self._walker if w not in excludes]
# endregion

class Main:
    def __init__(self, model_path: pathlib.Path, n_inputs: int, n_hidden: int, n_layers: int, lr: float, weight_decay: float, batch_size: int):
        # select device
        print(torch.version.cuda)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # was needed for debugging another model, might be helpful at some point
        # torch.autograd.set_detect_anomaly(True) 

        # create model and optimiser objects
        n_head = 1 # nb of heads for multiheaded attention
        self.model = PinyinTransformer(N_TOKENS, n_inputs, n_head, n_hidden, n_layers).double().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # reduce the learning after 20 epochs by a factor of 10
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        # load the model
        self.load_model(model_path)

        print(self.model)
        print('number of parameters: ', sum(
            [param.nelement() for param in self.model.parameters()]))

        # optimizer to device
        self.optim_to_device()

        # Create training and testing split of the data.

        # region OLD STUFF:
        # train_set = SubsetSC("training")
        # print("while generating sets")
        # test_set = SubsetSC("testing")
        # # validation_set = SubsetSC("validation")
        # # exit()
        # print("after generating sets")
        # waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
        # endregion

        print("generating index")
        data_handler = DataHandler(
            phone_set_path=pathlib.Path("../../data/AISHELL-3/phone-set.txt"),
            train_path=pathlib.Path("../../data/AISHELL-3/train/train_content.txt"),
            test_path=pathlib.Path("../../data/AISHELL-3/test/test_content.txt"),
            train_pickle=pathlib.Path("../../data/AISHELL-3/train_pickle_mini.pkl"),
            valid_pickle=pathlib.Path("../../data/AISHELL-3/valid_pickle_mini.pkl"),
            test_pickle=pathlib.Path("../../data/AISHELL-3/test_pickle_mini.pkl")
        )
        print("generating sets")
        train_set = data_handler.get_set("train")
        validation_set = data_handler.get_set("validate")
        # test_set = data_handler.get_set("test")
        _, sample_rate, _ = train_set[0]
        print("done generating sets")

        self.loss_criterion = nn.CrossEntropyLoss(ignore_index=data_handler.index.phon2idx[PADDING])
        # region sorting labels xd??
        # labels = []
        # try:
        #     with open("labels.txt") as file:
        #         labels = file.read().split()

        # except:
        #     labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
        #     with open("labels.txt", "w") as file:
        #         for l in labels:
        #             file.write(f"{l}\n")

        # print("after sorting labels")
        # endregion

        # loading waveform from .wav:
        #   waveform, sample_rate = torchaudio.load("audio_file.wav", normalize=True)
        new_sample_rate = 256 # 8000 # 16000 # original: 44100
        self.transform_resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        self.transform_mfcc = torchaudio.transforms.MFCC(sample_rate=new_sample_rate, n_mfcc=n_inputs, melkwargs={"n_mels": n_inputs}) # (sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 80, "n_mels": 23, "center": False}) # n_mfcc=13 (default:40), , log_mels=True

        self.transform_resample = self.transform_resample.double().to(self.device)
        self.transform_mfcc = self.transform_mfcc.double().to(self.device)

        if self.device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_handler.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.validate_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=data_handler.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        # self.test_loader = torch.utils.data.DataLoader(
        #     test_set,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     drop_last=False,
        #     collate_fn=data_handler.collate_fn,
        #     num_workers=num_workers,
        #     pin_memory=pin_memory,
        # )

        # TODO: unified loss_fn for training and testing as self.loss(..)
        # loss_fn = nn.CrossEntropyLoss()
        # loss_fn = F.nll_loss
        # loss_fn = nn.NLLLoss

        self.run()

    def load_model(self, model_path: pathlib.Path):
        try:
            checkpoint = torch.load(model_path) # str(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            print("loaded the model successfully")
        except:
            self.epoch = 0
            self.loss = float('inf')

    def optim_to_device(self):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)

    def loss_fn(self, output, target) -> torch.Tensor:
        """
        output: ( B, S, N ) - Batch size, Sequence length, N of tokens - sequence of probabilities for each token is generated
        target: ( B, S ) - Batch size, Sequens length - target is a sequence of tokens
        """
        # TODO: use nn.NLLLoss

        # CEL:  outputs B, N_TOKENS, d1,d2,...
        #       targets B, d1,d2,...

        # target size:
        # torch.Size([200, 27])
        # output:
        # torch.Size([200, 27, 3396])
        # RuntimeError: Expected target size [200, 3396], got [200, 27]
        return self.loss_criterion(output.transpose(-1, -2), target.long())

    def train(self, epoch: int, log_interval: int, pbar: tqdm, pbar_update: float):
        self.model.train()
        loss = 0
        # self.model.to(self.device) # model = ?

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            # TODO: input_lengths from train_loader https://github.com/foamliu/Speech-Transformer/blob/master/train.py#L109
            # ALWAYS feed target to the model, https://github.com/foamliu/Speech-Transformer/blob/master/test.py#L57

            # apply transform and model on whole batch directly on device
            for transform in [self.transform_resample, self.transform_mfcc]:
                print(f"input:\n{data.size()}")
                data = transform(data).to(self.device)
            # after transforms:
            # data.size() = (B, MFCC_FEATS, S) // Bach_size, Mfcc_features, Sequence_length (S is lower after downsampling)

            # print(f"shape of mfcc: {tuple(data.size())}")

            print(f"input:\n{data.size()}")
            data = data.transpose(-1, -2)
            print(f"input:\n{data.size()}")

            print(f"target size:\n{target.size()}")
            # """
            # TODO: fix tgt input: AssertionError: was expecting embedding dimension of 256, but got 26
            #       in TransformerDecoder (Embedding layer)
            #       ??GO LINEAR (B, tgtSEQ, 1) -> (B, tgtSEQ, 256)??
            # """

            output = self.model(data, target)
            print(f"output:\n{output.size()}")

            loss = self.loss_fn(output, target)
            print(f"loss:\n{loss}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #prevent gradient explosions
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            # print training stats
            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset):.4f} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # update progress bar
            pbar.update(pbar_update)

            raise Exception("abort")
            # region save
            # record loss
            # if losses[-1] < best_loss[-1]:
            #     os.system("rm *model_mfcc_best*")
            #     best_loss[-1] = losses[-1]
            #     model_path = f'GUR4_model_mfcc_best_v3_e{epoch_parameter}_l{int(100*losses[-1])}'
            #     # save(model.state_dict(), model_path)
            #     save({
            #         'epoch': epoch_parameter,
            #         'model_encoder_state_dict': model_encoder.state_dict(),
            #         'model_decoder_state_dict': model_decoder.state_dict(),
            #         'optimizer_enc_state_dict': optimizer_enc.state_dict(),
            #         'optimizer_dec_state_dict': optimizer_dec.state_dict(),
            #         'scheduler_enc_state_dict': scheduler_enc.state_dict(),
            #         'scheduler_dec_state_dict': scheduler_dec.state_dict(),
            #         'loss': loss,
            #         'best_loss': best_loss[-1]
            #         }, model_path)
            # endregion

        # self.losses.append(loss.item())
        new_model_path = f'pinyinTransformer-E{epoch}_l{int(100*loss.item())}'
        print(f"saving: {new_model_path}")
        save({
            'epoch': iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            }, new_model_path)

    def validate(self, epoch: int, pbar: tqdm, pbar_update):
        # TODO: use a validation set
        # invalid_num = 0
        # good_guess = 0
        # waveform_len, *_ = test_set[0]
        # waveform_len = transform_mfcc(waveform_len.to(device))
        # waveform_len = waveform_len.size(2)

        self.model.eval()
        correct = 0
        for data, target in self.validate_loader_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            for transform in [self.transform_resample, self.transform_mfcc]:
                data = transform(data).to(self.device)
            
            output = self.model(data)
            # get likely index
            pred = output.argmax(dim=-1)
            correct += pred.squeeze().eq(target).sum().item()

            pbar.update(pbar_update)
        
        print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(self.test_loader.dataset)} ({100. * correct / len(self.test_loader.dataset):.0f}%)\n")

    def run(self):
        log_interval = 20
        n_epoch = 25
        pbar_update = 1 / (len(self.train_loader) + len(self.validate_loader))
        with tqdm(total=n_epoch) as pbar:
            for epoch_iter in range(self.epoch + 1, self.epoch + n_epoch + 1):
                self.train(epoch_iter, log_interval, pbar, pbar_update)
                self.validate(epoch_iter, pbar, pbar_update)
                # TODO: current_loss = test
                # current_loss = float("inf")
                # print(f"epoch: {epoch_iter}, loss: {current_loss}")
                self.scheduler.step()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=lambda p: pathlib.Path(p).absolute(), dest="model_path",
                        help="optional path to previously saved model")
    parser.add_argument("--em-size", type=int, dest="n_inputs",
                        help="size of an embedding", default=256)
    parser.add_argument("--n-hidden", type=int, dest="n_hidden",
                        help="number of hidden layers", default=64)
    parser.add_argument("--n-layers", type=int, dest="n_layers",
                        help="number of encoder layers", default=2)
    parser.add_argument("--lr", type=float, dest="lr",
                        help="learning rate for training", default=0.01)
    parser.add_argument("--weight-decay", type=float, dest="weight_decay",
                        help="weight decay for training", default=0.0001)
    parser.add_argument("--batch-size", type=int, dest="batch_size",
                        help="batch size for training", default=200)
    args = parser.parse_args()

    Main(args.model_path, args.n_inputs, args.n_hidden, args.n_layers, args.lr, args.weight_decay, args.batch_size)