import argparse
from classifier_utils import train_process, LEARNING_RATE, HIDDEN_LAYER, EPOCHS


def get_input_args():
    parser = argparse.ArgumentParser(description='Train Flower Image Classification')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU or not')
    parser.add_argument('--arch', type=str, default='densenet121', help='architecture [available: alexnet, '
                                                                        'densenet121, vgg16]',
                        required=True)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=HIDDEN_LAYER, help='hidden units for fc layer')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--save_dir', type=str, help='directory to save checkpoints')
    parser.add_argument('--saved_model', type=str, default='my_checkpoint.pth',
                        help='path to the saved model checkpoint')
    return parser.parse_args()


def main():
    args = get_input_args()
    train_process(args)


if __name__ == "__main__":
    main()
