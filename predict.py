import torch
import argparse
import json
from classifier_utils import load_checkpoint, process_image, predict

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

    idx_to_class = {v: k for k, v in loaded_model.class_to_idx.items()}
    top_class, top_probability = predict(img, loaded_model, idx_to_class, topk=args.topk, use_gpu=args.gpu)

    # print('Predicted Classes: ', top_class)
    print('Predicted Class Names: ')
    [print(cat_to_name[x]) for x in top_class]
    print('Predicted Probabilities: ', top_probability)


if __name__ == "__main__":
    main()
