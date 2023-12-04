# README

Edit Pilot is a Visual Studio Code extension that features automatic code edit recommendations.

## Features

The plugin is composed of an **edit locator** and an **edit generator.** 

The edit locator, combining a discriminator model and a locator model, is for suggesting edit locations according to *previous edits* and *current commit message.*

The edit generator, based on a single generator model, is for generating replacements or additions somewhere in the code, from suggested locations or manually selected. It also needs *previous edits* and *current commit message* , together with the code to replace.

## Usage

1. Edit the code, as our extension will automatically record most previous edits.

2. To trigger `Predict Locations` or `Change Commit Message`, **right-click** anywhere in the editor, then click the action in the menu.

3. To trigger `Generate Edits`, select part of the code for **replacing** (or select none for **adding**) in the editor, then **right-click** and click `Generate Edits` in the menu.

4. After the model generates possible edits at that range, a difference tab with pop up for you to edit the code or switch to different edits. **There are buttons on the top right corner of the difference tab to accept, dismiss or switch among generated edits.**

## Deployment

This extension is currently not released in VS Code Extension Store. Follow the next steps to run the extension in development mode in VS Code.

### To get started

`git clone` this project to somewhere you want, then `cd` the project or simply open the project directory in VS Code.

### Run backend models

Our model scripts require **Python 3.10** and **Pytorch with CUDA.**  

#### Step 1: Install Python dependencies

> [!IMPORTANT]
> For *Windows* and *Linux using CUDA 11.8,* please follow [PyTorch official guide](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA before the following steps.

Using `pip` :

```shell
pip install -r requirements.txt
```

Or using `conda` :

```shell
conda create -n code-edit
conda activate code-edit
conda install python=3.10.13
python -m pip install -r requirements.txt
```


#### Step 2: Download models into the project directory

Download `models.zip` from [here](https://drive.google.com/file/d/1nW1NCeelOUZfqebrncKvlB7FVZutjQsT/view?usp=sharing), unzip it, then put the `models` folder into the project root directory.

#### Step 3: Start the backend

Simply run `server.py` from the project root directory

```shell
python src/model_server/server.py
```

> [!NOTE]
> Always remember to start up backend models which the extension must base on.

### Run extension in debugging mode

#### Step 1: Install Node.js

See [Node.js official website](https://nodejs.org/en/download).

#### Step 2: Install Node dependencies

In the project root directory, install Node packages

```shell
npm install
```

#### Step 3: Run extension using VS Code development host

Open the project directory in VS Code if didn't, press `F5`, then choose `Run Extension` if you are required to choose a configuration. Note that other extensions will be disabled in the development host.

## Issues

The project is still in development. Not fully tested on different platforms.

**Enjoy!**
