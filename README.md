# yet-another-titanic-project
<div id="top"></div>


<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="https://github.com/diegommezp28/yet-another-titanic-project/assets/47110686/e1dd81d9-b693-424d-aa34-322738669ec1" alt="Logo" width="1000" height="250">
  </a>

  <p align="center">
  Simple yet feature-rich training pipeline
  
  </br>
    <a href="https://github.com/diegommezp28/yet-another-titanic-project/issues">Report Bug</a>
    
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About</a>
      <ul>
        <li><a href="#built-with">Built with</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Intro</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Use</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About
This project implements a simple package for training and evaluating a model for the famous titanic dataset from kaggle 
https://www.kaggle.com/competitions/titanic/overview


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Python](https://www.python.org/)
* [Scikit Learn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org/)
* [Typer](https://typer.tiangolo.com/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Intro

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation
The whole package is inside the [titanicTraining](https://github.com/diegommezp28/yet-another-titanic-project/tree/main/titanicTraining) folder. So, after cloning the project, you can install it with:

```
pip install titanicTraining
```

This will install in your environment an script named `titanic`, which is a CLI app made with [Typer](https://typer.tiangolo.com/). 

This package DOES NOT include the dataset, you can dowload it from [here](https://www.kaggle.com/competitions/titanic/overview) or, better yet, use the [kaggle](https://pypi.org/project/kaggle/) package to download it. For that, run:

```
pip install kaggle
```

```
kaggle competitions download -c titanic
```


**Important:** Remember that in order to use the kaggle package you will need to configure your kaggle.json credentials locally, see [here](https://github.com/Kaggle/kaggle-api#api-credentials) an explanation. 

## Usage

Since this is made with Typer, you can see all the options by running

```
titanic --help
```

There are 3 main commands: `run`, `resume`, `predict`. 

#### run

* All the info in: `titanic run --help`

This is the main command which runs the whole training pipeline. You can see all the command options by running `titanic run --help`.
The main input (optional, with a default path) is the .yaml file which has all the training configs for the model and most important, it has the default route to the .zip file with the titanic dataset, you shall change this route if your local dataset does not match the default path. 
This repo already ships with the `titanic_train.yaml` config file that the script will look for, you can also specify a custom config file path with the `--config-file` option. 

```
titanic run
```
Will run the training pipeline. Automatically, it will save checkpoints of each step of the pipeline. The default checkpointing folder is at `titanic_train.yaml` in `base_runs_folder: runs`.

#### resume

* All the info in: `titanic resume --help`

A key feature of the package is the ability to resume the pipeline run from a previously saved checkpoint. You can resume from a .ckpt file by running:

```
titanic resume --ckpt-file <path_to_file>
```
#### predict

* All the info in: `titanic predict --help`

This is the command that lets you run predictions on new data with a previously trained pipeline. You do not need to specify the pipeline, by default it will look for the latest evaluation checkpoint inside the `runs` folder. Of course, if you want, you can provide a valid checkpoint using the `--ckpt-file` option and an output path to write the result using the `--output-path` option.

```
titanic predict <path_to_data>
```

## Docker

There is already a Dockerfile in this repo that will configure everything you need to run this code, including downloading the data and setting up a volume for such data folder. Just take into account when building the image that docker will look for the `kaggle.json` file in the main folder. You can use the `sample.kaggle.json`, rename it and fill the neccesary information inside of it, or just download it from kaggle following the instructions from [here](https://github.com/Kaggle/kaggle-api#api-credentials). The entrypoint of the Docker image will already be the `titanic` script, so you just have to add the relevant options and arguments. 

**Tip:** Run the container with the `-t` flag to have a colored output in the console. i.e.:

```
docker run -t --rm <image_name>:<image_tag> run
```

## Features

Here are a list of features that the `titanic` script has:
* Running training pipeline from zip data
* Running training pipeline from train and test csv files
* Configuring the RandomForest hyperparameters from a .yaml file
* Automatic checkpoints for each stage of the pipeline (Ingest, Preprocessing, Training, Evaluate)
* Resuming pipeline from a saved  checkpoint
* Perform prediction on new data using a previously train model or an Evaluate Checkpoint
* Perform prediction on preprocessed and feature enriched data
* Save prediction in an output csv file


<!-- LICENSE -->
## License

Distributed under the APACHE License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Diego Gómez  - diego_gomezpolo@outlook.com - [LinkedIn](https://www.linkedin.com/in/diegomezp28/)

Project Link: [https://github.com/diegommezp28/yet-another-titanic-project](https://github.com/diegommezp28/yet-another-titanic-project)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/diegommezp28/yet-another-titanic-project.svg?style=for-the-badge
[contributors-url]: https://github.com/diegommezp28/yet-another-titanic-project/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/diegommezp28/yet-another-titanic-project.svg?style=for-the-badge
[forks-url]: https://github.com/diegommezp28/yet-another-titanic-project/network/members
[stars-shield]: https://img.shields.io/github/stars/diegommezp28/yet-another-titanic-project.svg?style=for-the-badge
[stars-url]: https://github.com/diegommezp28/yet-another-titanic-project/stargazers
[issues-shield]: https://img.shields.io/github/issues/diegommezp28/yet-another-titanic-project.svg?style=for-the-badge
[issues-url]: https://github.com/diegommezp28/yet-another-titanic-project/issues
[license-shield]: https://img.shields.io/github/license/diegommezp28/yet-another-titanic-project.svg?style=for-the-badge
[license-url]: https://github.com/diegommezp28/yet-another-titanic-project/blob/master/LICENSE
[product-screenshot]: images/screenshot.png
