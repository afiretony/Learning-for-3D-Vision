import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, render_pcd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    print('model created!')
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))
    sigma = 0
    test_data += torch.normal(mean=0, std = sigma * torch.ones_like(test_data))
    print(test_data[0].max())
    print(test_data[0].min())
    # ------ TO DO: Make Prediction ------
    batch_size = 4
    num_batches = test_data.shape[0] // batch_size
    correct = 0
    pred_labels = torch.zeros_like(test_label)

    for i in tqdm(range(num_batches)):
        pred_results = model(test_data[i * batch_size : (i+1) * batch_size].to(args.device))
        pred_label = torch.argmax(pred_results, -1, keepdim=False).cpu()
        pred_labels[i * batch_size : (i+1) * batch_size] = pred_label
        # Compute Accuracy
        correct+= pred_label.eq(test_label[i * batch_size : (i+1) * batch_size].data).cpu().sum().item()
        
    test_accuracy = correct / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))
    cm = confusion_matrix(test_label, pred_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=None)
    fig, ax = plt.subplots(figsize=(5, 2.7))
    disp.plot(ax=ax)
    plt.savefig('output/confusion_m.png')
    # plt.saveas


    

    # render fail examples

    # for i in range(test_data.shape[0]):
    #     pred_result = model(test_data[i].view(1,-1,3).to(args.device))
    #     pred_label = torch.argmax(pred_result, -1, keepdim=False).cpu()
    #     if pred_label.cpu() != test_label[i]:
    #         print(i)
    #         render_pcd(test_data[i], 'q1_fail_'+str(i), args.device, int(pred_label), int(test_label[i]))

    # render successful examples
    # ids = np.random.choice(test_data.shape[0], 10, replace=False)
    
    # ids = [17, 189, 408, 429, 489, 539, 557, 561, 890, 932]
    # for i in ids:
    #     pred_result = model(test_data[i].view(1,-1,3).to(args.device))
    #     pred_label = torch.argmax(pred_result, -1, keepdim=False).cpu()
    #     if pred_label.cpu() == test_label[i]:
    #         print('Rendering:', i)
    #         render_pcd(test_data[i], 'q1_success_'+str(args.num_points) +'_'+ str(i), args.device, int(pred_label), int(test_label[i]))
    
    # i = 539
    # pred_result = model(test_data[i].view(1,-1,3).to(args.device))
    # pred_label = torch.argmax(pred_result, -1, keepdim=False).cpu()
    # render_pcd(test_data[i], 'q1_noise_{}'.format(sigma), args.device, int(pred_label), int(test_label[i]))


