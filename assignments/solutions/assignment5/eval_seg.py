import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg
from tqdm import tqdm

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)

    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])
    # add noise
    sigma = 1
    test_data += torch.normal(mean=0, std = sigma * torch.ones_like(test_data))
    
    # ------ TO DO: Make Prediction ------
    # with torch.no_grad():
    batch_size = 16
    num_batches = test_data.shape[0] // batch_size
    print('---------------------')
    # print(num_batches)

    correct = 0
    pred_labels = torch.zeros_like(test_label)

    for i in tqdm(range(num_batches)):
        pred_results = model(test_data[i * batch_size : (i+1) * batch_size].to(args.device))
        pred_label = torch.argmax(pred_results, -1, keepdim=False).cpu()
        pred_labels[i * batch_size : (i+1) * batch_size] = pred_label

        # Compute Accuracy
        correct += pred_label.eq(test_label[i * batch_size : (i+1) * batch_size].data).sum()
        
    test_accuracy = correct / (test_data.shape[0]* test_data.shape[1])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)

    # for ii in range(10):
    #     i = ii * 50
    #     viz_seg(test_data[i], test_label[i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
    #     viz_seg(test_data[i], pred_labels[i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)

    # viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, args.i), args.device)
    # viz_seg(test_data[args.i], pred_labels[args.i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, args.i), args.device)
