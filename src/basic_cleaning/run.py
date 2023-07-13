#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results to Weights and Biases
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"

import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact

    logger.info('Downloading artifact from weights and biases')
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info('Cleaning data')
    idx = df.price.between(args.min_price, args.max_price)
    df = df[idx]


    logger.info('Saved dataframe to disk')
    df.to_csv('clean_data.csv' ,index=False)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file('clean_data.csv')

    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")


    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Atrifact from Weights and Biases",
        required=True,
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name to be used for data uploaded to Weights and Biases",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type of artifact to be uploaded",
        required=False,
        default="raw_data",
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact to be uploaded",
        required=False,
        default="Preprocessed data from the original data sources"
    )


    parser.add_argument(
        "--min_price",
        type=float,
        help="minimum price for the data",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="maximum price for the data",
        required=True
    )


    args = parser.parse_args()

    go(args)
