"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

import pandas as pd
import argparse

from models.model.sit import SignalTransformer
from utils.dataloader import Dataloader


class ModelSelectionPipeline():
    def __init__(self):
        pass

    def load_data(self, test_mode=False):
        dataloader = Dataloader(
            raw_data_path='data/raw',
            window_data_path='data/TSB_512',
            feature_data_path=None,
            split_file='data/splits/supervised/split_TSB.csv'
        )

        train_set, val_set, test_set = dataloader.create_splits()

        if test_mode:
            train_set = train_set[:10]
            val_set = val_set[:5]
            test_set = test_set[:5]

        train_datasets = set([x.split('/')[0] for x in train_set])
        val_datasets = set([x.split('/')[0] for x in val_set])
        test_datasets = set([x.split('/')[0] for x in test_set])
        all_datasets = train_datasets | val_datasets | test_datasets

        all_windowed_data = dataloader.load_multiple_window_timeseries_parallel(all_datasets)

        set_dicts = []
        for curr_set in [train_set, val_set, test_set]:
            print(curr_set)
            x, y, fnames = dataloader.load_timeseries_parallel(curr_set)

            set_dicts.append({
                'x': x,
                'y': y,
                'names': fnames,
                'windowed': all_windowed_data[all_windowed_data.index.str.contains('|'.join(fnames))]
            })
            
        return set_dicts
    
    def model_creation(self):
        return SignalTransformer(
            dropout_rate=0.2,
            attn_dropout_rate=0.2,
            use_conv_stem=True,
            use_conv_stem_original=True,
            hidden_dims=[16, 32, 64]
        )

    def train(self):
        pass

    def evaluate(self):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    
    pipeline = ModelSelectionPipeline()
    model = pipeline.model_creation()
    train_data, val_data, test_data = pipeline.load_data(test_mode=args.test)
    print(train_data)

    pipeline.train()
    pipeline.evaluate()