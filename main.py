import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()
    src_path = os.path.join(root_path, "src")

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            ##################
            # basic cleaning
            ##################
            _ = mlflow.run(
                os.path.join(src_path, "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": f"{config['main']['project_name']}/sample.csv:latest" ,
                    "artifact_name": config['etl']['preprocess_name'],
                    "artifact_type": "raw_data",
                    "artifact_description": "preprocessed data",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            ##################
            # data check #
            ##################
            _ = mlflow.run(
            os.path.join(src_path, "data_check"),
                "main",
                parameters={
                    "ref": f"{config['main']['project_name']}/{config['etl']['preprocess_name']}:reference",
                    "csv": f"{config['main']['project_name']}/{config['etl']['preprocess_name']}:latest",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            ##################
            # data_split     #
            ##################
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                version='main',
                parameters={
                    "input": f"{config['main']['project_name']}/{config['etl']['preprocess_name']}:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
                os.path.join(src_path, "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": f"{config['main']['project_name']}/{config['etl']['preprocess_name']}:latest",
                    "rf_config": rf_config,
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify": config["modeling"]["stratify_by"],
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_model"
                },
            )

        if "test_regression_model" in active_steps:

            ##################
            #  test model #
            ##################
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                version='main',
                parameters={
                    "mlflow_model": f"{config['main']['project_name']}/random_forest_model:prod",
                    "test_dataset": f"{config['main']['project_name']}/test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    go()
