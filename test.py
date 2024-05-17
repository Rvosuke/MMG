import argparse
from tqdm import tqdm
import torch


class Trainer:
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': 0, 'pin_memory': True}
        self.val_loader = make_data_loader(args, **kwargs)

        # specify the device to use
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load entiry model to cuda if available
        model = torch.load('../models/model_save.pth', map_location=device)
        model.eval()
        self.model = model

    def validation(self):
        tbar = tqdm(self.val_loader, desc='\r')
        for sample in tbar:
            image, target_clinical = sample['image'], sample['label']
            clinical = target_clinical[:, 5:9]
            if self.args.cuda:
                image, clinical = image.cuda(), clinical.cuda()
            with torch.no_grad():
                output = self.model(image, clinical)

            output = output.data.cpu()
            output = torch.argmax(output, 1).numpy()[0]
            print(output)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--data', type=str, default='shufa_4', choices=['shufa_4'], help='data name (default: pascal)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]

    trainer = Trainer(args)
    trainer.validation()


if __name__ == "__main__":
    main()
