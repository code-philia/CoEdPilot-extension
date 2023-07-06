# README

本插件为提供自动代码推荐的 VSCode 插件。

## 功能
插件能够根据前一次修改和提交的 commit message，通过高亮推荐修改位置。用户点击或选中高亮位置会给出一个或多个修改建议。若采纳修改建议，用户可以通过点击修改建议实现自动修改。

修改位置推荐功能拥有以下触发方式：
* 修改内容识别（编辑器内容变化且光标所在行发生改变）；
    1. 键盘输入/删除内容，完成后鼠标点击另一行；
    2. 键盘输入内容后回车换行；
    3. 删除键删除本行直到返回上一行；
    4. 复制多行内容到任意位置；
    5. 选中单行内容并删除/键入修改，完成后鼠标点击另一行；
    6. 向前选中删除多行内容，完成后鼠标点击另一行；
    7. 向后选中删除多行内容；
    7. 向前选中替换相同行数内容；
    8. 向后选中替换相同行数内容，完成后鼠标点击另一行；
    9. 向前/后选中替换不同行数的内容；
* 提交 commit message；
* 接受插件给出的修改内容建议。

## 插件用法

1. 插件暂未发布，请在 VSCode 内按 **F5**，在 debug 模式内使用；
2. 提交 commit message，请在编辑器内任意位置单击右键，在菜单内选择 **Enter commit messge**。此时顶部会出现输入框。请在完成输入后按 **Enter** 键确认;
3. 若要关闭 commit message 输入框，请点击后按 **Esc**;
4. 关闭 commit message 输入框不会删除当前保留的 commit message，若要更新 commit message，请在 commit message 输入框内输入新的内容并按 **Enter** 键确认；
5. 当出现粉红色高亮的推荐修改位置时，用户可以点击或选择一个位置，此时会在位置前方出现**蓝色小灯泡**。点击该灯泡即可查看多个推荐的修改内容；
6. 若接受推荐的修改内容，用户可以直接点击实现修改。

## 开发者操作
1. 请修改 src/extension.js 开头的 Hyper-parameters，包括 高亮效果的设置（fontcolor，bgcolor）和 后端 python 脚本路径（pyPathEditRange，pyPathEditContent）；
5. 本插件拥有两个后端 Python 脚本，分别为：**修改位置预测脚本** 和 **修改内容预测脚本**，其路径应分别记录在 pyPathEditRange 和 pyPathEditContent 两个参数中；
6. src/range_model.py 和 src/content_model.py 主要实现了对传入传出参数的转化处理，请用实际模型替换两个脚本中的 RangeModel() 和 ContentModel() 两个函数。

## 问题

* 暂未为真实数据的例子实现前端演示或前后端的联合演示；
* src/edit.json 和 src/demo_model.py 是为了真实数据例子 test/test case/code-demo.go 的前端演示所准备的数据。

**Enjoy!**
