import torch
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from transformers import (SegformerImageProcessor,
                          SegformerForSemanticSegmentation)

from utils import parse_args, get_optimiser
from data.dataset import SegFormerDataset


def get_data(batch_size, image_size, device):
    train_ratio = 0.7

    torch.manual_seed(42)
    transform = T.Compose([
        T.ToTensor()
    ])
    processor = SegformerImageProcessor(do_reduce_labels=False)
    full_dataset = SegFormerDataset(transform, image_size, device, processor)
    train_len = int(len(full_dataset) * train_ratio)
    val_test_len = len(full_dataset) - train_len
    val_len = val_test_len // 2
    test_len = val_test_len - val_len

    train_dataset, val_test_dataset = random_split(
        full_dataset, [train_len, val_test_len])
    val_dataset, test_dataset = random_split(
        val_test_dataset, [val_len, test_len])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def get_model():
    id2label = {
        0: "background",
        1: "object",
    }

    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                             num_labels=2,
                                                             id2label=id2label,
                                                             label2id=label2id)

    return model


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    num_epochs = args.num_epochs
    save_path = args.save_path
    patience = args.patience

    train_dataloader, val_dataloader, test_dataloader = get_data(
        args.batch_size, args.image_size, device)
    model = get_model()
    optimiser = get_optimiser(args, model.parameters())
