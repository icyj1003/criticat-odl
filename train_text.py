import argparse

import torch
from torch.utils.data import DataLoader, random_split

from customdataset import TestingDataset
from module import BertEmbedding, BiLSTM, TextCNN, TokenEmbedding
from train_tools import eval_offline, train_offline
from utils import ensure_dir_exists, set_all_seeds

parser = argparse.ArgumentParser()


def create_model(args):
    # load G
    if args.g_model == "train":
        G = TokenEmbedding(
            max_length=args.max_length,
            embedding_dim=args.embed_dim,
            max_features=args.max_vocab_size,
            device=args.device,
        ).to(args.device)
    elif args.g_model == "bert":
        G = BertEmbedding(max_length=args.max_length, device=args.device).to(
            args.device
        )

    # load F
    if args.f_model == "bilstm":
        F = BiLSTM(
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embed_dim,
            dropout_rate=args.dropout,
        ).to(args.device)
    elif args.f_model == "textcnn":
        F = TextCNN(
            embedding_dim=args.embed_dim,
            num_filters=args.num_filters,
            filter_sizes=args.filter_sizes,
            dropout_rate=args.dropout,
        ).to(args.device)

    return G, F


def load_dataset(dir):
    dataset = torch.load(dir)
    train_dataset = TestingDataset(
        [
            {k: v for k, v in d.items() if k in ["post_message", "label"]}
            for d in dataset["train"].data
        ]
    )
    test_dataset = TestingDataset(
        [
            {k: v for k, v in d.items() if k in ["post_message", "label"]}
            for d in dataset["test"].data
        ]
    )

    dev_dataset = TestingDataset(
        [
            {k: v for k, v in d.items() if k in ["post_message", "label"]}
            for d in dataset["dev"].data
        ]
    )

    del dataset

    return train_dataset, test_dataset, dev_dataset


def online(args: argparse.Namespace):
    pass


def offline(args: argparse.Namespace):
    # load dataset
    train_dataset, test_dataset, dev_dataset = load_dataset(args.dataset_dir)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # create model
    G, F = create_model(args)

    # optimizer and loss function
    optimizer = torch.optim.Adam(
        params=list(G.parameters()) + list(F.parameters()), lr=args.lr
    )
    loss_function = torch.nn.CrossEntropyLoss()

    # train
    for epoch in range(1, args.epochs + 1):
        (
            train_loss,
            train_acc,
            train_f1,
            train_precision,
            train_recall,
            train_proba,
        ) = train_offline(
            dataloader=train_loader,
            G=G,
            F=F,
            loss_function=loss_function,
            optimizer=optimizer,
            device=args.device,
        )

        dev_loss, dev_acc, dev_f1, dev_precision, dev_recall, dev_proba = eval_offline(
            dataloader=dev_loader,
            G=G,
            F=F,
            loss_function=loss_function,
            device=args.device,
        )

        if args.verbose:
            print(
                f"epoch: {epoch:02} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | train_f1: {train_f1:.4f} | dev_loss {dev_loss:.4f} | dev_acc: {dev_acc:.4f} | dev_f1: {dev_f1:.4f}"
            )
    # eval
    if args.verbose:
        print(
            eval_offline(
                dataloader=test_loader,
                G=G,
                F=F,
                loss_function=loss_function,
                device=args.device,
            )
        )


if __name__ == "__main__":
    parser.add_argument(
        "--online", default=False, action="store_true", help="Trigger online training"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the offline dataset directory",
        default="D:\\storage\\odl\\cache\\offline\\reintel2020\\dataset.pt",
    )
    parser.add_argument(
        "--sessions_dir",
        type=str,
        help="Path to the online dataset directory",
        default="D:\\storage\\odl\\cache\\online_session\\reintel2020\\",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the model directory",
        default="D:\\storage\\odl\\checkpoints\\reintel2020\\",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=1)
    parser.add_argument(
        "--max_length", type=int, help="Max sentence length", default=65
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--g_model", type=str, help="Embedding layers", default="train")
    parser.add_argument("--f_model", type=str, help="Plastic layers", default="bilstm")
    parser.add_argument(
        "--embed_dim", type=int, help="Embedding dimension", default=300
    )
    parser.add_argument(
        "--max_vocab_size", type=int, help="Max vocabulary size", default=100000
    )
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension", default=512)
    parser.add_argument(
        "--num_filters", type=int, help="Number of filters", default=100
    )
    parser.add_argument("--dropout", type=float, help="Dropout", default=0.5)
    parser.add_argument(
        "--filter_sizes", help="Filter sizes", nargs="+", type=int, default=[3, 4, 5]
    )
    parser.add_argument("--device", type=str, help="Device", default="cuda:0")
    parser.add_argument("--seed", type=int, help="Seed", default=42)
    parser.add_argument("--verbose", action="store_true", help="Verbose", default=False)
    parser.add_argument("--early_stoping", action="store_true", help="Verbose", default=False)

    args = parser.parse_args()

    # testing
    assert args.g_model in [
        "train",
        "bert",
    ], "Embedding layer must be either train or bert"
    assert args.f_model in ["bilstm", "textcnn"], "Layer type not supported"
    assert len(args.filter_sizes) == 3, "Number of filter sizes must be 3"
    if args.g_model == "bert":
        assert args.embed_dim == 768, "Embedding dimension must be 300 for bert model"

    # set seed
    set_all_seeds(args.seed)

    # training
    if args.online:
        online(args)
    else:
        offline(args)
