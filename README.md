# ✏️ CoEdPilot

CoEdPilot is a Visual Studio Code extension that features automatic code edit recommendations.

## 🚀 Demo
> [!NOTE]
> Please click the image to watch the demo video on YouTube.

<div align="center">
   <a href="https://youtu.be/6G2-7Gf0Fhc">
   <img src="./media/demo_cover.png" width="600" />
   </a>
</div>

## ⚙️ Functionality

The extension introduces two major features: **Edit Locator** and **Edit Generator.** 

### Edit Locator

Combining a **🔍 file locator (discriminator) model** and a **🎯 line locator model.** It suggests edit locations according to *previous edits* and *current edit description.*

### Edit Generator

Based on a single **📝 generator model.** It generates replacements or insertions somewhere in the code, from suggested locations or manually selected. It also requires *previous edits* and *current edit description* and, in addition, the code to replace.

## ✨ UI

### Overview

![Overview](media/ui1.png)

+ Predicted locations will be displayed as a tree view in the left ⬅️ and also highlighted in the active editor
+ Query status will be displayed in the status bar ↘️
+ Edit description is accepted in the input above ⬆️

### Diff View

![Diff View](media/ui2.png)

Once performing a prediction on a line, a diff view is shown for switching ↔️ or editing ✏️ the prediction result.

## 🧑‍💻Usage

1. Edit the code, as our extension will automatically record most previous edits.

2. Run `Predict Locations`: **right-click** anywhere in the editor and select it in the menu, or use the default keybinding `Ctrl + Alt + L` (in MacOS `Cmd + Alt + L`)

3. Run `Generate Edits`: select the code to be edited in the editor, then **right-click** and select it in the menu, or use the default keybinding `Ctrl + Alt + E` (in MacOS `Cmd + Alt + E`)

> [!NOTE]
> To select code for editing, you can:
>   * Click recommended locations in the left location list;
>   * Select part of the code for **replacing**;
>   * Select nothing to generate **insertion** code at the cursor position.
>
> And by default accepting an edit will trigger another location prediction immediately (you can change this in extension configuration).

4. Manually `Change Edit Description`: **right-click** and select it in the menu. By default the input box will automatically show at query **whenever the edit description is empty.**


4. After the model generates possible edits at that range, a difference tab with pop up for you to switch to different edits or edit the code. **There are buttons on the top right corner of the difference tab to accept, dismiss or switch among generated edits.**


## 🕹️ Run Extension

This extension is currently not released in VS Code Extension Store. Follow the next steps to run the extension in development mode in VS Code.


### 1. Run backend models

> [!NOTE]
> Always remember to start up backend models which the extension must base on.

Our model scripts require **Python 3.10** and **Pytorch with CUDA.**  

#### Step 1: Install Python dependencies

> [!IMPORTANT]
> For *Windows* and *Linux using CUDA 11.8,* please follow [PyTorch official guide](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA before the following steps.

Using `pip` (with Python 3.10):

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

As mentioned before, we respectively prepared 3 models (*file locator*(including embedding model, dependency analyzer and a regression model), *line locator*, and *generator*) for each language. Supported languages are `go`, `python`, `java`, `typescript` and `javascript`.

1. Download and rename **models for different languages** and **dependency analyzer** from [Huggingface Collections](https://huggingface.co/collections/code-philia/coedpilot-65ee9df1b5e3b11755547205). 
    * `dependency-analyzer/`: dependency anaylzer model, available in [Huggingface](https://huggingface.co/code-philia/dependency-analyzer);
    * `embedding_model.bin`: embedding model for file locator, available in [Huggingface](https://huggingface.co/code-philia/CoEdPilot-file-locator);
    * `reg_model.pickle`: , linear regression model, available in [Huggingface](https://huggingface.co/code-philia/CoEdPilot-file-locator);
    * `locator_model.bin`: model for line locator, available in [Huggingface](https://huggingface.co/code-philia/CoEdPilot-line-locator);
    * `generator_model.bin`: model for generator, available in [Huggingface](https://huggingface.co/code-philia/CoEdPilot-generator).

2. To deploy models for one language, put its unzipped model folder **named with the language**.
    ```
    edit-pilot/
        models/
            dependency-analyzer/
            {language}/
                embedding_model.bin
                reg_model.pickle
                locator_model.bin
                generator_model.bin
    ```

#### Step 3: Start the backend

Simply run `server.py` from the project root directory

```shell
python src/model_server/server.py
```

### 2. Run extension

#### Step 1: Install Node dependencies

In the project root directory, install Node packages

```shell
npm install
```

> [!NOTE]
> Require Node.js (version >= 16). If Node.js not installed, please follow [Node.js official website](https://nodejs.org/en/download) to intall.

#### Step 2: Run extension using VS Code development host

Open the project directory in VS Code if didn't, press `F5`, then choose `Run Extension` if you are required to choose a configuration. Note that other extensions will be disabled in the development host.

## 🛠️ Advanced Deployment

We recommend to try this extension with backend deployed locally. This will require **CUDA** and **~4GB** video memory. But deploying backend remotely is also easy, since key is to match extension configuration of VS Code and the server listening configuration. 

By default `server.py` fetches configuration from `server.ini` then listens to `0.0.0.0:5001`. The extension client sends requests to `coEdPilot.queryUrl`, by default `http://localhost:5001`.

For basic remote backend deployment:

+ On the backend machine, confirm the IP on LAN/WAN, and open up a port through firewall.
+ Use that port in `server.ini` then run the backend `server.py`.
+ When using the extension, set `coEdPilot.queryUrl` in VS Code settings (press `Ctrl + ,` then search) to the proper connectable IP of the server.

## ❓ Issues

The project is still in development, not fully tested on different platforms. 

Welcome to propose issues or contribute to the code.

**😄 Enjoy coding!**
