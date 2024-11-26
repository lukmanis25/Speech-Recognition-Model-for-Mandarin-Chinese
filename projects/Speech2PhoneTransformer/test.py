import pathlib
from argparse import ArgumentParser

import torch

from data import DataHandler

class Main:
    def __init__(self, batch_size, num_workers):
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

        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=data_handler.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

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