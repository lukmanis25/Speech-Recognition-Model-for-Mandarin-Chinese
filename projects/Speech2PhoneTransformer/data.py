from pathlib import Path
import pickle
from typing import Literal
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


START_OF_SEQUENCE = "<SOS>"
END_OF_SEQUENCE = "<EOS>"
PADDING = "<PAD>"


class PhonemeIndex:
    def __init__(self, phone_set_path: Path):
        """
        creates a phone dictionary from AISHELL-3 phone-set.txt

        example line in phone_set:

        zhi3	1065
        """
        assert phone_set_path.exists()
        self.phon2idx = {START_OF_SEQUENCE: 0, END_OF_SEQUENCE: 1, PADDING: 2}
        self.idx2phon = [START_OF_SEQUENCE, END_OF_SEQUENCE, PADDING]

        lines = [line for line in phone_set_path.read_text().splitlines() if line and not line.startswith("#")]
        for line in lines:
            phone = line.split()[0]
            if not phone in self.phon2idx:
                self.idx2phon.append(phone)
                self.phon2idx[phone] = len(self.idx2phon) - 1


class DataHandler:
    def __init__(self, phone_set_path: Path, train_path: Path, test_path: Path,
                 train_pickle: Path=None, valid_pickle: Path=None, test_pickle: Path=None):
        # Login using e.g. `huggingface-cli login` to access this dataset
        self.index = PhonemeIndex(phone_set_path)
        train_labels, train_wavs = self.tokenize(train_path)
        test_labels, test_wavs = self.tokenize(test_path)
        labels = train_labels + test_labels
        wavs = train_wavs + test_wavs
        self.wav2seq = dict(zip(wavs, labels))
        self.pickles = {
            "train": train_pickle,
            "validate": valid_pickle,
            "test": test_pickle
        }

    def get_set(self, set_name: Literal["train", "validate", "test"]):
        """
        AISHELL-3 splits: train - 8222, test - 3982

        add split: valid - 25% of train

        output: [(waveform, sample_rate, label(.wav filename)), ..]

        TODO: save dataset in an intermediate format (tensor.save(), pickle.dump/load)
        """
        pickle_name = self.pickles.get(set_name)
        ret: list
        if pickle_name and Path(pickle_name).exists():
            # load a pickled dataset split
            with open(pickle_name, "rb") as file:
                ret = pickle.load(file)
        else:
            # load a dataset split with load_dataset
            if set_name == "test":
                data_set = load_dataset("shenberg1/aishell3", split=set_name)["audio"]
            
            else:
                # manually split train split to (training, validation)
                data_set = load_dataset("shenberg1/aishell3", split="train")["audio"]
                df = pd.DataFrame(data_set)
                df = np.split(df.sample(frac=1, random_state=0), [int(.75*len(df))])[0 if set_name == "train" else 1]
                data_set = df.to_dict('records')
            
            ret = [(record["array"], record["sampling_rate"], Path(record["path"]).name) for record in data_set]
            if pickle_name:
                # pickle the dataset
                with open(pickle_name, "wb") as file:
                    pickle.dump(ret, file)

        return ret


    def collate_fn(self, batch: list[tuple[list[int],int,str]]):
        """
        batch: [(waveform, sample_rate, label(.wav filename)), ..]
        
        output: waveform tensors, targets (token lists)
        """
        # TODO: A data tuple has the form:
        # waveform, sample_rate, labelSEQ

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label in batch:
            tensors += [torch.tensor(waveform)]
            targets += [torch.tensor(self.wav2seq[label]).type(torch.int64)]

        # Group the list of tensors into a batched tensor
        # print("tensors.size():")
        # print([tsr.size() for tsr in tensors[:7]])
        # print("targets.size():")
        # print([tgt.size() for tgt in targets[:7]])

        # tensors.size():
        # [118784, 74373, 82908, 318175, 102477, 87210, 98789]
        # targets.size():
        # [15, 6, 7, 21, 11, 6, 10]

        tensors = pad_sequence(tensors, batch_first=True).double()
        targets = pad_sequence(targets, batch_first=True, padding_value=self.index.phon2idx[PADDING]).int()

        # print("\napplied padding\n")
        # print("tensors.size():")
        # print([tsr.size() for tsr in tensors[:7]])
        # print("targets.size():")
        # print([tgt.size() for tgt in targets[:7]])
        return tensors, targets


    def tokenize(self, content_path: Path) -> tuple[list[list[int]], list[str]]:
        """
        tokenize aishell-3 content.txt files
        
        example line: SSB06930002.wav	武 wu3 术 shu4 始 shi3 终 zhong1 被 bei4 看 kan4 作 zuo4 我 wo3 国 guo2 的 de5 国 guo2 粹 cui4
        
        outputs: a Tensor of size (N, T) where N is the number of sequences and T is length of the longet sequence

                 and a list of .wav filenames in order of the tokenized sequences
        """
        assert content_path.exists()

        ids = []
        wavs = []
        lines = [line for line in content_path.read_text(encoding="utf-8").splitlines() if line]
        for line in lines:
            tokens = [START_OF_SEQUENCE] + [token for token in line.split()[1:] if token.isascii()] + [END_OF_SEQUENCE]
            line_ids = [self.index.phon2idx[token] for token in tokens]
            # ids.append(torch.tensor(line_ids).type(torch.int64))
            ids.append(line_ids)
            wavs.append(line.split()[0])

        # return pad_sequence(ids, batch_first=True, padding_value=self.index.phon2idx[PADDING]), wavs
        return ids, wavs
