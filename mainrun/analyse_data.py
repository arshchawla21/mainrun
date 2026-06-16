"""
Analyse the distribution of the NL data,
to inform tokeniser design.

changelog:
  import key functions from train (16/06)
"""

import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path
from datasets import load_dataset

from train import get_titles, Hyperparameters

def main():
    args = Hyperparameters()
    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    