import torch
import argparse
import json
from classifier_utils import load_checkpoint, process_image, predict, extract_classes

use_gpu = torch.cuda.is_available


def main():
    parser = argparse.ArgumentParser(description='Flower Classification Predictor')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')
    parser.add_argument('--image_path', type=str, help='path oto image')
    parser.add_argument('--saved_model', type=str, default='my_checkpoint.pth',
                        help='path to your saved model checkpoint')
    parser.add_argument('--mapper_json', type=str, default='cat_to_name.json',
                        help='path to category to name mapper')
    parser.add_argument('--topk', type=int, default=5, help='display top k classes probabilities')

    args = parser.parse_args()

    with open(args.mapper_json, 'r') as f:
        cat_to_name = json.load(f)

    loaded_model, loaded_optimizer, epochs = load_checkpoint(args.saved_model)
    img = args.image_path
    img = process_image(img)
    print(img.shape)

    probabilities_tensor = predict(args.image_path, loaded_model, topk=args.topk, use_gpu=args.gpu)
    classes, probabilities = extract_classes(cat_to_name, probabilities_tensor)

    print('Predicted Class Names: ', classes)
    print('Predicted Probabilities: ', probabilities)


if __name__ == "__main__":
    main()
