# README

This plugin is a VSCode extension that provides automatic code edit recommendations.

## Feature
The plugin can highlight recommended edit locations based on the previous changes and user's description of the edit. When the user clicks or selects a highlighted position, one or more modification suggestions will be presented. If the user accepts the suggestions, they can apply the modifications automatically by clicking on the suggestion.

The edit location recommendation feature can be triggered in the following ways:
* Detection of content modification (when the editor content changes and the cursor moves to a new line):
    1. Typing or deleting content and then clicking on another line when done.
    2. Typing content and pressing enter to create a new line.
    3. Pressing the delete key to remove content until reaching the previous line.
    4. Copying multiple lines of content to any location.
    5. Selecting a single line of content, making changes, and then clicking on another line when done.
    6. Selecting forward and deleting multiple lines of content and then clicking on another line when done.
    7. Selecting backward and deleting multiple lines of content.
    8. Selecting forward and replacing with content of the same number of lines.
    9. Selecting forward or backward, then replacing with content of different numbers of lines.
* Enter edit description.
* Accepting modification suggestions provided by the plugin。

## Plugin Usage

1. The plugin is not yet published, please use it in debug mode by pressing **F5** in VSCode.
2. To enter edit description, right-click anywhere in the editor, select **Enter edit description** from the menu. An input box will appear at the top. After entering the description, press **Enter** to confirm.
3. To close the edit description input box, click it and then press **ESC**.
4. Closing the edit description input box does not empty the current edit description. To update the edit description, enter new content in the input box and press **Enter** to confirm.
5. Red highlight suggests deleting or updating this line, while green highlight suggests adding content after this line.
6. When highlighted edit locations appear, the user can click or select a position. A blue dot will appear at the right side of the location. Clicking the blue dot will display multiple recommended modification contents.
If the user accepts the recommended modifications, they can directly click it to implement the changes.

## Developer's Operations
1. Please modify the Hyper-parameters at the beginning of *src/extension.js*, including settings for highlighting effects (font color, background color) and the paths to the backend Python scripts (*pyPathEditRange, pyPathEditContent*).
2. Please modify variable *model_name* in both *src/range_model.py* and *src/content_model.py*. 
3. All path should use the absolute path.
4. *src/range_model.py* bridges the front-end and the back-end to provide edit location recommendation.
5. *src/content_model.py* bridges the front-end and the back-end to provide edit content recommendation.
6. The large langauge models are *src/locator_pytorch_model.bin* and *src/generator_pytorch_model.bin*.

## Issues

* Currently, the Python scripts must be located locally and use stdin and stdout for content transmission.


**Enjoy!**
