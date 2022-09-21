import os
import torch
from argparse import ArgumentParser

def compute_statistics(params):
    dataset = params.dataset.lower()
    # default folder is in the same directory
    save_path = "/home/vietnguyen/hiwi/DGM_pytorch/stats"
    file_path = os.path.join(save_path,f"{params.dataset}_test_acc.txt".lower())
    with open(file_path, "r") as f:
        test_acc = [float(line.strip()) for line in f.readlines()]
    tensor = torch.tensor(test_acc)

    with open(file_path, "a+") as f:
        f.write(f"mean: {str(tensor.mean().item())}\n")
        f.write(f"std: {str(tensor.std().item())}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='Cora')
    params = parser.parse_args()

    compute_statistics(params)