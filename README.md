# README

Edit Pilot is a Visual Studio Code extension that features automatic code edit recommendations.

## Features

The plugin is composed of an **edit locator** and an **edit generator.** 

The edit locator, combining a discriminator model and a locator model, is for suggesting edit locations according to *previous edits* and *current commit message.*

The edit generator, based on a single generator model, is for generating replacements or additions somewhere in the code, from suggested locations or manually selected. It also needs *previous edits* and *current commit message* , together with the code to replace.

## Usage

1. Edit the code, as our extension will automatically record most previous edits.

2. To trigger `Predict Locations` or `Change Commit Message`, right-click anywhere in the editor, then click the action in the menu.

3. To trigger `Generate Edits`, select part of the code for replacing (or select none for adding) in the editor, then right-click and click `Generate Edits` in the menu.

4. After the model generates possible edits at that range, a difference tab with pop up for you to edit the code or switch to different edits. **There are buttons on the top right corner of the difference tab to accept, dismiss or switch among generate edits.**

## Deployment

This extension is currently not released in VS Code Extension Store. Follow the next steps to run the extension in development mode in VS Code.

### To get started

`git clone` this project to somewhere you want, then `cd` the project or simply open the project directory in VS Code.

### Run backend models

Our model scripts require **Python 3.10** and **Pytorch with CUDA.**  

#### Step 1: Install Python dependencies

Using `pip` :

```shell
pip install torch torchvision torchaudio transformers retriv flask tqdm bleu
```

Or using `conda` :

```shell
conda create -n code-edit
conda activate code-edit
conda install python=3.10.13
python -m pip install torch torchvision torchaudio transformers retriv flask tqdm bleu
```

> [!IMPORTANT]
> For Windows, follow [PyTorch official guide](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA. 

#### Step 2: Download models into the project directory

Download `models.zip` from here, unzip it, then put the `models` folder into the project root directory.

#### Step 3: Start the 

Simply run `python src/model_server/server.py` from the project root directory.

### Run extension in debugging mode

1. Install [Node.js](https://nodejs.org/en/download).

2. In the project directory, run `npm install` .

3. Open the project directory in VS Code if didn't. Press `F5` and choose `Run Extension` if you are required to choose a configuration. Note that other extension will be disabled in the development host.

> [!Note]
> Remember to start up backend models before running the extension.

## Issues

The project is still in development. Not fully tested on different platforms.

**Enjoy!**
