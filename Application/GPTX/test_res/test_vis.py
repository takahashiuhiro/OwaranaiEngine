import matplotlib.pyplot as plt

# 读取数据
file_path = 'GPT2_l1_nh32_ne1536.train.log.oe'  # 替换为你的文件路径
train_epochs = []
train_cur_values = []
valid_epochs = []
valid_cur_values = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.split()
        if line.startswith('train:Epoch:'):
            epoch = int(parts[1])
            cur_value = float(parts[3])
            train_epochs.append(epoch)
            train_cur_values.append(cur_value)
        elif line.startswith('valid:Epoch:'):
            epoch = int(parts[1])
            cur_value = float(parts[3])
            valid_epochs.append(epoch)
            valid_cur_values.append(cur_value)
        else:
            params_ = line.split(":")[1]

# 绘制图像
plt.figure(figsize=(10, 5))
plt.plot(train_epochs, train_cur_values, linestyle='-', color='b', label='train')
plt.plot(valid_epochs, valid_cur_values, linestyle='-', color='y', label='valid')

# 添加标签和标题
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress: train/valid loss, params:'+params_)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像
plt.savefig('training_progress.png')
plt.show()
