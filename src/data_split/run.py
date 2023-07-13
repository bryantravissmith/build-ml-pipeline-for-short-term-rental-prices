#!/usr/bin/env python
"""
Downloads artifact, splits into training and test sets, then uploads to Weights and Biases
"""
import argparse
import logging
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="data_split")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Splits data to train and test data")


    parser.add_argument(
        "--input_artifact", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--artifact_root", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--test_size", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
