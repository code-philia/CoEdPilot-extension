# README

This plugin is a VSCode extension that provides automatic code edit recommendations.

## Feature
The plugin is composed of an edit locator and an edit generator. The edit locator can highlight recommended edit locations based on previous edits and the edit description provided by user. When the user clicks or selects a highlighted location, multiple edit suggestions will be provided by the edit generator. If the user chooses to accept the edit suggestion, they can click on the suggestion, and the plugin will automatically apply the edit.

The edit location recommendation feature can be triggered in the following ways:
* Content edit (editor content changes, and the cursor moves to a different line):
    1. Typing/deleting content and then clicking on another line.
    2. Typing content and pressing "Enter" for a new line.
    3. Pressing the "Backspace" key to delete the current line until returning to the previous line.
    4. Copying multiple lines of content to any location.
    5. Selecting a single line of content, making edits (deleting/typing), and then clicking on another line.
    6. Selecting multiple lines of content forward, deleting, and then clicking on another line.
    7. Selecting multiple lines of content backward and deleting them.
    8. Selecting and replacing content with the same number of lines forward.
    9. Selecting and replacing content with the same number of lines backward, then clicking on another line.
    10. Selecting and replacing content with different numbers of lines forward/backward.
* User enter edit description.
* Accept edit suggestion provided by the extension.

## Plugin Usage
1. The plugin is not yet published. Please press `F5` within VS Code to use it in debug mode.
2. To submit an edit description, right-click anywhere in the editor, then select **Enter edit description** from the menu. An input box will appear at the top. After entering your message, press `Enter` to confirm.
3. To close the edit description input box, click it and then press `Enter`.
4. Closing the edit description input box will not delete the current saved edit description. If you want to update the edit description, enter the new content inside the edit description input box and press `Enter` to confirm.
5. Red highlighting indicates recommended edits for the current line, while green highlighting suggests additions to the code after the current line.
6. When recommended edit locations are highlighted, users can click or select a location, and a **blue dot** will appear in front of it. Clicking on it will display multiple recommended edit options.
7. If you want to accept a recommended edit, you can directly click on it to apply the change.

## Deployment of the extension
1. Install [Node.js](https://nodejs.org/en/download).
2. Install packages required for VS Code extension: 
```
npm install -g yo generator-code
```
3. Download extension [code](https://github.com/code-philia/Code-Edit).
4. Download [backend models](https://drive.google.com/file/d/1MYn68MOJsUQLDwYNedINZtxgWcXDiLJG/view?usp=sharing).
5. Rename the edit locator model as *locator_pytorch_model.bin*, and the edit generation model as *generator_pytorch_model.bin*, move them to folder *src/*.
6. Create environment and install dependencies of the backend models. The backend model can also work with cuda.
```
conda create -n code-edit
conda activate code-edit
pip install transformers==4.13.0
pip3 install torch torchvision torchaudio
```
7. Switch to the extension directory, then install node modules.
```
cd /path/to/edit-pilot
npm install
```
8. Open file *src/extension.js*, edit the path variable `PyInterpreter` with the absolute path of the interpreter. E.g.: */home/username/miniconda3/envs/codebert/bin/python*.
9. Open the extension folder within VS Code, open *src/extension.js*, press `F5` to run the extension in debug mode. If a VS Code menu pop up, select "VS Code Extension".
10. New VS Code window should appear and the extension is ready. 

## Issues

* Currently, Python scripts must be located locally and use stdin and stdout for content exchange.

**Enjoy!**
