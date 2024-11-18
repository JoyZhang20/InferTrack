import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图形样式
sns.set(style="whitegrid")

# 读取Excel文件
df = pd.read_excel("../data/infer_dataset.xlsx",sheet_name="yolov8")  # 替换为实际文件路径

# 绘制 data_size 列的箱型图
plt.figure(figsize=(8, 6))
sns.boxplot(data=df['data_size'])
plt.title("Boxplot of Data Size")
plt.xlabel("Data Size")
plt.savefig("../result_plot/data_size.png", dpi=300)
plt.show()

# 绘制 infer_time_s 和 infer_time_m 对比的箱型图
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['infer_time_s', 'infer_time_m', 'infer_time_l']])
plt.title("Comparison of Inference Time")
plt.ylabel("Infer Time")
plt.xticks([0,  1,  2], ['Yolov8s', 'Yolov8m', 'Yolov8l'])
plt.savefig("../result_plot/infer_time.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['accuracy_s', 'accuracy_m', 'accuracy_l']])
plt.title("Comparison of Inference Accuracy")
plt.ylabel("Infer Accuracy")
plt.xticks([0,  1,  2], ['Yolov8s', 'Yolov8m', 'Yolov8l'])
plt.savefig("../result_plot/infer_acc.png", dpi=300)
plt.show()
