import argparse
import sys
import torch


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--training", default="yes", type=str, 
                        help="Do we train?")
    
    # transformer model
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, 
                        help="transformer model")

    # Number of training epochs (authors recommend between 2 and 4)
    parser.add_argument("--num_epochs", default=4, type=int, 
                        help="Number of training epochs (recommended 2-4).")

    parser.add_argument("--batch_size", default=32, type=int, 
                        help="Recommended 16 or 32.")
    
    parser.add_argument("--freeze_layer_count", default=-1, type=int, 
                        help="")
    
    parser.add_argument("--data_path", default="UD_French-GSD", type=str, 
                        help="")
    
    
    args = parser.parse_args()

    return args