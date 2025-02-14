import pandas as pd
import numpy as np
import random


def get_sample_data_freq(new_df, is_category,is_final):  
    
    filtered_df = new_df[new_df['is_category'] == is_category]
    if is_final:
        columns_to_merge = [f'question{i}' for i in range(1, 2)]
    else:
        columns_to_merge = [f'question{i}' for i in range(1, 7)]
 
    merged_series = filtered_df[columns_to_merge].melt(value_name='question')['question']
 
    merged_series = merged_series.dropna()

    # 统计频率
    frequency_counts = merged_series.value_counts()

    # 将频率结果转换为 DataFrame
    sorted_frequency_counts = frequency_counts.sort_index()

    #print("频率统计结果：")
    #print(sorted_frequency_counts)
    
    return sorted_frequency_counts


def get_known_ids_with_counts(row, is_category, counts,cognition_to_id,):
    if is_category == 1: 
        known_category = [col for col in row.index if col.startswith('CPC') and row[col] == 1]
        known_ids = [cognition_to_id.get(category, category) for category in known_category]
    else: 
        known_products = [col for col in row.index if col.startswith('PAU') and row[col] == 1]
        known_ids = [cognition_to_id.get(product, product) for product in known_products]
    known_ids_with_counts = [(id, counts.get(id, 0) if is_category else counts.get(id, 0)) for id in known_ids]

 
    sorted_known_ids_with_counts = sorted(known_ids_with_counts, key=lambda x: x[1], reverse=False)
    
    ''' 
    randomized_ids_with_counts = []
    current_count = None
    same_count_ids = []
    for id, count in sorted_known_ids_with_counts:
        if count != current_count:
            if same_count_ids:
                randomized_ids_with_counts.extend(random.sample(same_count_ids, len(same_count_ids)))
                same_count_ids = []
            randomized_ids_with_counts.append((id, count))
            current_count = count
        else:
            same_count_ids.append((id, count))
    if same_count_ids:
        randomized_ids_with_counts.extend(random.sample(same_count_ids, len(same_count_ids)))

    return randomized_ids_with_counts
    '''
    return sorted_known_ids_with_counts


file_path = 'template.xlsx'

xls = pd.ExcelFile(file_path)

sheet_names = xls.sheet_names

print("Sheets in the Excel file:", sheet_names)

all_data = {}
for sheet in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    all_data[sheet] = df

df_raw_data = all_data['raw data'].iloc[0:]
df_category_label = all_data['category label']
df_brand_label = all_data['Brand label']

df_category_target_EMO = all_data['category目标值']
df_product_target_EMO = all_data['product目标值']
df_category_target_FUN = all_data['category目标值2']
df_product_target_FUN = all_data['product目标值2']


df_raw_data_sorted = df_raw_data.sort_values(by='serial')

df_category_label_id = df_category_label['Perception Label']
df_brand_label_id = df_brand_label['Perception Label'][:80]

cognition_to_category_id = df_category_label.set_index('认知变量')['Perception Label'].to_dict()
cognition_to_product_id = df_brand_label.set_index('认知变量')['Perception Label'].to_dict()

print(cognition_to_category_id)
print(cognition_to_product_id)

new_df = pd.DataFrame()

columns = ['Serial']
columns.append(f'is_category') 

columns.append(f'set') 

# question1 就是final question
for i in range(1, 7):
    columns.append(f'question{i}') 

for i in range(1, 7): 
    for j in range(1, 29):
        columns.append(f'question{i} EMOattri{j}') # question1 EMO1attri1
    for j in range(1, 30):
        columns.append(f'question{i} FUNattri{j}')

new_df = new_df.reindex(columns=columns)

new_df['Serial'] = df_raw_data_sorted['serial']
new_df['set'] = df_raw_data_sorted['attriset']
new_df['is_category'] = df_raw_data_sorted['GROUPCATE'] 
new_df['question1'] = df_raw_data_sorted.apply(lambda row: row['CATEGORY'] if row['GROUPCATE'] == 1 else row['PRODUCT'], axis=1)


# 打印结果
print("Final category Consumed的统计信息 ")
category_counts = get_sample_data_freq(new_df, 1,True)
print(category_counts)

print("\nFinal product Consumed的统计信息 ")
product_counts = get_sample_data_freq(new_df, 0,True)
print(product_counts)

new_df['known_ids_with_counts'] = df_raw_data_sorted.apply(lambda row: get_known_ids_with_counts(row, row['GROUPCATE'],category_counts if row['GROUPCATE'] == 1 else product_counts,cognition_to_category_id if row['GROUPCATE'] == 1 else cognition_to_product_id ), axis=1)

for i in range(2, 7):
    new_df[f'question{i}'] = new_df['known_ids_with_counts'].apply(lambda x: x[i-2][0] if len(x) >= i else None)


# 将 'question1' 到 'question6' 的列值放入 'questions' 列中
new_df['questions'] = new_df[[f'question{i}' for i in range(1, 7)]].apply(lambda row: row.tolist(), axis=1)

# 打印新数据框的 'questions' 列，以检查结果
print(new_df[['Serial', 'questions']].head())


# 打印结果
print("Final category Consumed的统计信息 ")
final_category_freq = get_sample_data_freq(new_df, 1, True)
print(final_category_freq)

print("\nFinal product Consumed的统计信息 ")
final_product_freq = get_sample_data_freq(new_df, 0, True)
print(final_product_freq)

print("\nAll category Consumed的统计信息 ")
all_category_freq = get_sample_data_freq(new_df, 1, False)
print(all_category_freq)

print("\nAll product Consumed的统计信息 ")
all_product_freq = get_sample_data_freq(new_df, 0, False)
print(all_product_freq)

# 将结果保存到文件
with pd.ExcelWriter('frequencies.xlsx') as writer:
    final_category_freq.to_excel(writer, sheet_name='Final Category', index=True)
    all_category_freq.to_excel(writer, sheet_name='All Category', index=True)
    final_product_freq.to_excel(writer, sheet_name='Final Product', index=True)
    all_product_freq.to_excel(writer, sheet_name='All Product', index=True)

print("Frequencies saved to frequencies.xlsx")

# 分配

def get_rows_with_question(data_df, set_value, is_category_value, question_id):
    # 用于存储满足条件的行的结果
    matched_rows = []
    
    # 遍历每一行
    for index, row in data_df.iterrows():
        # 检查 set 和 is_category 是否匹配，并且 questions 列包含 question_id
        if row['set'] == set_value and row['is_category'] == is_category_value:
            # 如果 'questions' 列包含 question_id
            if question_id in row['questions']:
                # 获取 question_id 在 questions 列中的位置
                question_position = row['questions'].index(question_id)
                # 将 (Serial, question_position) 作为元组添加到 matched_rows
                matched_rows.append((index,(row['Serial'], question_position)))
    
    return matched_rows


def set_random_values(data_df, matched_rows, p,attri_num,isEMO):
    for index, (serial, question_position) in matched_rows:
        # 动态构造目标列名：question{i} EMOattri{j} question6 FUNattri27
        column_name = f'question{question_position+1} {"EMO" if isEMO else "FUN"}attri{attri_num}'
        
        # 生成一个 [0, 1) 范围内的随机数
        rand_value = random.random()
        
        # 根据概率 p 设置值为 1 或 0
        if rand_value <= p:
            data_df.at[index, column_name] = 1  # 设置为 1
        else:
            data_df.at[index, column_name] = 0  # 设置为 0
    
    return data_df


def process_and_update_df(df_target, new_df,isEMO,is_category_value):
    print(f"{is_category_value}  {isEMO} " )  
    for index, row in df_target.iterrows():
        attr_id = row['id']
        print(f"{is_category_value}  {isEMO}  {attr_id} ")
        # 遍历每一列，从第三列开始
        for column in df_target.columns[2:]:
            target_id = column
            target_value = row[column]
            
            # 打印目标信息
            #print(f"{attr_id}  {target_id}: {target_value}", end=" | ")
            
            # 获取满足条件的行
            matched_results = get_rows_with_question(new_df, 1 if attr_id % 2 == 1 else 2, is_category_value, target_id)
            
            # 使用随机数填充数据
            new_df = set_random_values(new_df, matched_results, target_value / 100.0, attr_id, isEMO)
        
    
         
    print()  # 换行
    
    return new_df



new_df = process_and_update_df(df_category_target_EMO, new_df,True,1)
 
new_df = process_and_update_df(df_category_target_FUN, new_df,False,1)
new_df = process_and_update_df(df_product_target_EMO, new_df,True,0)
new_df = process_and_update_df(df_product_target_FUN, new_df,False,0)


output_file_path = 'new_data.xlsx'
print(f"Try to saved to {output_file_path}")
new_df.to_excel(output_file_path, index=False)

print(f"Data saved to {output_file_path}")