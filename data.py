import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"C:\Users\16171\Desktop\data1.xlsx"  # 确保文件路径正确
df = pd.read_excel(file_path)

columns = df.columns
prefix_map = {}

for col in columns:
    if col.endswith('_pairs'):
        prefix = col.replace('_pairs', '')
        prefix_map.setdefault(prefix, {})['pairs'] = col
    elif col.endswith('_answers'):
        prefix = col.replace('_answers', '')
        prefix_map.setdefault(prefix, {})['answers'] = col

valid_pairs = []

for prefix, pair in prefix_map.items():
    if 'pairs' in pair and 'answers' in pair:  
        valid_pairs.append({
            'prefix': prefix,
            'pairs': pair['pairs'],
            'answers': pair['answers']
        })

valid_pairs_set = {}

for index, row in df.iterrows():
    for item in valid_pairs:
        prefix = item['prefix']
        pairs = row[item['pairs']]
        answers = row[item['answers']]
        if pairs is None or answers is None or not isinstance(pairs, str) or not isinstance(answers, str):
            break

        all_pairs = pairs.split("#-#")
        all_answers = answers.split(",")

        for number in range(0, len(all_pairs)):
            pair = all_pairs[number]
            answer = int(all_answers[number].replace("_", ""))
            
            all_pair_set = pair.replace("_", "").split(",")
            pairs_tuple = tuple(sorted([int(x) for x in all_pair_set]))

            if pairs_tuple not in valid_pairs_set:
                valid_pairs_set[pairs_tuple] = {}

            if answer not in valid_pairs_set[pairs_tuple]:
                valid_pairs_set[pairs_tuple][answer] = 0

            valid_pairs_set[pairs_tuple][answer] += 1

output_data = []

for pairs_tuple, answers_data in sorted(valid_pairs_set.items()):
    total_count = sum(answers_data.values())  
    for answer, count in answers_data.items():
        percentage = (count / total_count) * 100  
        output_data.append({
            'Pairs Tuple': str(pairs_tuple),
            'Answer': answer,
            'Count': count,
            'Percentage': f"{percentage:.2f}%"
        })

# 生成 DataFrame
output_df = pd.DataFrame(output_data)


# 拆分 "Pairs Tuple" 列，提取成对的 Item A 和 Item B
output_df[['Item A', 'Item B']] = output_df['Pairs Tuple'].str.extract(r'\((\d+), (\d+)\)')
output_df['Item A'] = output_df['Item A'].astype(int)
output_df['Item B'] = output_df['Item B'].astype(int)

# 将 "Percentage" 列转换为数值
output_df['Percentage'] = output_df['Percentage'].str.rstrip('%').astype(float)

# 获取唯一的 Item 列表
items = sorted(set(output_df['Item A']).union(set(output_df['Item B'])))

# 创建空的 DataFrame 作为比较矩阵
comparison_matrix = pd.DataFrame(index=items, columns=items, data=0.0)

# 填充矩阵
for _, row in output_df.iterrows():
    item_a = row['Item A']
    item_b = row['Item B']
    percentage = row['Percentage']

    # 矩阵中的 (A, B) 位置填充 A 相对于 B 的胜率
    comparison_matrix.loc[item_a, item_b] = percentage

    # B 的胜率 = 100% - A 的胜率
    comparison_matrix.loc[item_b, item_a] = 100 - percentage


output_excel_path = "pairwise_comparison_matrix.xlsx"
comparison_matrix.to_excel(output_excel_path, index=True)
print(f"Excel 文件已保存: {output_excel_path}")


plt.figure(figsize=(10, 8))
sns.heatmap(comparison_matrix, annot=True, cmap="coolwarm", fmt=".1f")

# 添加标题和标签
plt.title("Pairwise Comparison Matrix")
plt.xlabel("Item")
plt.ylabel("Item")

# 保存图片
output_image_path = "pairwise_comparison_matrix.png"
plt.savefig(output_image_path)

print(f"热力图已保存: {output_image_path}")

# 显示图表
plt.show()
