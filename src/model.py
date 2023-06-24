# This Python script is purly for test, the actural script is a LLM
import re
import sys
import random

def process_text(text):
    words = ['fly', 'happy', 'good', 'car']
    word_positions = []

    for word in words:
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.finditer(pattern, text)
        for match in matches:
            if word == 'fly':
                replace = 'dive'
            elif word == 'happy':
                replace = 'sad'
            elif word == 'good':
                replace = 'bad'
            else:
                replace = 'bike'
            word_positions.append([match.start(), match.end(),replace])

    return {'data': word_positions}

# # 读取从 Node.js 传递的文本
input_text = sys.stdin.read()

# 处理文本并生成修改三元组
modifications = process_text(input_text)

# 将修改三元组作为输出发送给 Node.js
print(modifications)
sys.stdout.flush()

# if __name__ == '__main__':
#     sentence = """function fly(happy) {
#     let good = 3;
#     car(good);
# }"""
#     modif = process_text(sentence)
#     print(modif)