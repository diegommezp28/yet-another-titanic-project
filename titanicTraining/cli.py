import typer
from typing import Optional
from typing_extensions import Annotated
from rich import print
import os
import yaml
import pathlib
from src import TrainModelPipeline

DEFAULT_CONFIG_FILE_PATH = "./titanic_train.yaml"

app = typer.Typer()


@app.command()
def run(
    config_file: str = typer.Option(
        DEFAULT_CONFIG_FILE_PATH, help="Path to config file"
    )
):
    """
    Run the whole training pipeline
    """
    if os.path.isfile(config_file):
        if pathlib.Path(config_file).suffix in [".yaml", ".yml"]:
            doc = open(config_file, "r")
            data = yaml.load(doc, Loader=yaml.Loader)
            trainer_kwargs = data["trainer_args"]
            zip_path = data["data_zip"]
            base_runs_folder = (
                data["base_runs_folder"] if "base_runs_folder" in data else "runs"
            )
            model_ckpt_name = (
                data["model_ckpt_name"]
                if "model_ckpt_name" in data
                else "train_pipeline"
            )
            pipe = TrainModelPipeline(
                zip_path=zip_path,
                base_runs_folder=base_runs_folder,
                model_ckpt_name=model_ckpt_name,
                **trainer_kwargs,
            )
            pipe.run()
        else:
            print(
                f"[bold red]Error:[/bold red] The file at [blue]{config_file}[/blue] is not a .yaml or .yml file."
            )
            raise typer.Exit()
    else:
        print(
            f"[bold red]Error:[/bold red] There is no file in the specified path: [blue]{config_file}[/blue]"
        )
        raise typer.Exit()


@app.command()
def resume(
    ckpt_file: str = typer.Option("", help="Path to checkpoint file."),
    reload_configs: bool = typer.Option(
        True,
        help="Whether or not to re read the config file and change the pipeline values before resuming",
    ),
    config_file: str = typer.Option(
        DEFAULT_CONFIG_FILE_PATH, help="Path to config file"
    ),
):
    """
    Resume the training pipeline from the specified step.
    By default, this command will re read the config file and change the pipeline values before resuming.
    """
    if os.path.isfile(ckpt_file):
        if pathlib.Path(ckpt_file).suffix == ".ckpt":
            pipe = TrainModelPipeline.load(ckpt_file)

            if reload_configs:
                if os.path.isfile(config_file):
                    if pathlib.Path(config_file).suffix in [".yaml", ".yml"]:
                        doc = open(config_file, "r")
                        data = yaml.load(doc, Loader=yaml.Loader)
                        trainer_kwargs = data["trainer_args"]
                        zip_path = data["data_zip"]
                        base_runs_folder = (
                            data["base_runs_folder"]
                            if "base_runs_folder" in data
                            else "runs"
                        )
                        model_ckpt_name = (
                            data["model_ckpt_name"]
                            if "model_ckpt_name" in data
                            else "train_pipeline"
                        )
                        pipe.zip_path = zip_path
                        pipe.base_runs_folder = base_runs_folder
                        pipe.model_ckpt_name = model_ckpt_name
                        pipe.trainer_kwargs = trainer_kwargs
                    else:
                        print(
                            f"[bold red]Error:[/bold red] The file at [blue]{config_file}[/blue] is not a .yaml or .yml file."
                        )
                        raise typer.Exit()
                else:
                    print(
                        f"[bold red]Error:[/bold red] There is no file in the specified path: [blue]{config_file}[/blue]"
                    )
                    raise typer.Exit()

            pipe.resume()
        else:
            print(
                f"[bold red]Error:[/bold red] The file at [blue]{ckpt_file}[/blue] is not a .ckpt file."
            )
            raise typer.Exit()

    else:
        print(
            f"[bold red]Error:[/bold red] There is no file in the specified path: [blue]{ckpt_file}[/blue]"
        )
        raise typer.Exit()


@app.command()
def predict(
    data_path: str,
    processed: bool = typer.Option(
        False,
        help="Whether or nor the data is already processed and with features or is raw data.",
    ),
    output_path: str = typer.Option(
        "", help="Optional path to save the predictions results"
    ),
    pipeline_cktp: str = typer.Option(
        "",
        help="Optional path to the TrainerPipeline Checkpoint. By default this will look for the last cktp in the ./runs folder",
    ),
):
    """
    Use a previously trained pipeline to make predictions on new data
    """
    pass


# @app.command()
# def tests():
#     """
#     Run tests
#     """
#     pass


if __name__ == "__main__":
    app()
