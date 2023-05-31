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
    if os.path.isfile(config_file):
        if pathlib.Path(config_file).suffix in [".yaml", ".yml"]:
            doc = open(config_file, "r")
            data = yaml.load(doc, Loader=yaml.Loader)
            trainer_kwargs = data["trainer_args"]
            zip_path = data["data_zip"]
            pipe = TrainModelPipeline(zip_path=zip_path, **trainer_kwargs)
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
def resume():
    pass


if __name__ == "__main__":
    app()
