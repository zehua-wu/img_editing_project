# import pandas as pd
# import numpy as np
# import os

# def calculate_task_means(input_file, output_file='calculated_means.csv'):
#     print(f"正在读取文件: {input_file} ...")
    
#     # 1. 读取CSV
#     # dtype={'file_id': str} 非常重要，确保 '000...' 不会被解析成数字 0
#     try:
#         df = pd.read_csv(input_file, dtype={'file_id': str})
#     except FileNotFoundError:
#         print("错误：找不到输入文件。请检查路径。")
#         return

#     # 2. 预处理
#     # 确保 file_id 列存在
#     if 'file_id' not in df.columns:
#         # 如果第一列没有header，可能需要调整 read_csv 参数，这里假设第一列名为 file_id
#         print("错误：CSV中未找到 'file_id' 列。请确保列名正确。")
#         return

#     # 剔除 file_id 列，只保留数值列用于计算 mean
#     # 假设除了 file_id 其他都是 metrics
#     metric_columns = [col for col in df.columns if col != 'file_id']
    
#     # 将 NaN 填充为 0 (根据需求)
#     # 注意：只对数值列填充，避免报错
#     df[metric_columns] = df[metric_columns].fillna(0)

#     # 用于存储结果的列表
#     results = []

#     print("数据预处理完成，开始计算分组平均值...")

#     # ---------------------------------------------------------
#     # 3. 处理 Task 0 (整体任务)
#     # 逻辑：file_id 以 '0' 开头
#     # ---------------------------------------------------------
#     task0_data = df[df['file_id'].str.startswith('0', na=False)]
    
#     if not task0_data.empty:
#         # 计算平均值
#         means = task0_data[metric_columns].mean()
#         # 转换为字典并添加标签
#         row = means.to_dict()
#         row['Group_Label'] = 'Task_0_Overall'
#         results.append(row)
#         print(f"Task 0 (Overall): 计算完成，包含 {len(task0_data)} 行数据。")
#     else:
#         print("Task 0 (Overall): 未找到以 '0' 开头的数据。")

#     # ---------------------------------------------------------
#     # 4. 处理 Task 1-9 (细分任务)
#     # 逻辑：前三位 index (Task + Major + Minor)
#     # ---------------------------------------------------------
#     for task_id in range(1, 10):       # 1 到 9
#         for major_id in range(1, 3):   # 1 到 2 (2大类)
#             for minor_id in range(1, 5): # 1 到 4 (4小类)
                
#                 # 构建前缀，例如 "111", "112" ... "924"
#                 prefix = f"{task_id}{major_id}{minor_id}"
                
#                 # 筛选数据
#                 subset = df[df['file_id'].str.startswith(prefix, na=False)]
                
#                 if not subset.empty:
#                     # 计算平均值
#                     means = subset[metric_columns].mean()
#                     row = means.to_dict()
                    
#                     # 格式化标签名称，例如 Task_1_Major_1_Minor_1
#                     label_name = f"Task_{task_id}_Major_{major_id}_Minor_{minor_id}"
#                     row['Group_Label'] = label_name
                    
#                     results.append(row)
#                 else:
#                     # 如果这组没有数据，可以选择跳过，或者填入全是0的行
#                     # 这里选择跳过不写入CSV，如果需要占位，可以取消下面注释
#                     # print(f"前缀 {prefix} 没有匹配的数据，跳过。")
#                     pass

#     # ---------------------------------------------------------
#     # 5. 写入结果
#     # ---------------------------------------------------------
#     if results:
#         # 创建结果 DataFrame
#         result_df = pd.DataFrame(results)
        
#         # 将 Group_Label 移动到第一列
#         cols = ['Group_Label'] + [c for c in result_df.columns if c != 'Group_Label']
#         result_df = result_df[cols]
        
#         # 保存 CSV
#         result_df.to_csv(output_file, index=False)
#         print(f"\n成功！结果已保存至: {output_file}")
#         print(f"共生成了 {len(result_df)} 行汇总数据。")
#     else:
#         print("\n没有生成任何结果数据。")

# # ==========================================
# # 示例用法
# # ==========================================
# if __name__ == "__main__":
#     # 在这里修改你的输入文件名
#     INPUT_CSV = './evaluation_result.csv' 
    
#     # 这是一个生成测试数据的函数，如果你没有文件，可以先运行下面这一行生成假数据
#     # generate_dummy_data(INPUT_CSV) 
    
#     # 运行主逻辑
#     if os.path.exists(INPUT_CSV):
#         calculate_task_means(INPUT_CSV)
#     else:
#         print(f"未找到文件 {INPUT_CSV}，请修改脚本中的文件名或将文件放入同级目录。")




import pandas as pd
import numpy as np
import os
import argparse

def calculate_task_means(input_file, output_file='calculated_means.csv'):
    print(f"正在读取文件: {input_file} ...")
    
    # 1. 读取CSV
    # dtype={'file_id': str} 非常重要，确保 '000...' 不会被解析成数字 0
    try:
        df = pd.read_csv(input_file, dtype={'file_id': str})
    except FileNotFoundError:
        print("错误：找不到输入文件。请检查路径。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 2. 预处理
    # 确保 file_id 列存在
    if 'file_id' not in df.columns:
        # 如果第一列没有header，可能需要调整 read_csv 参数，这里假设第一列名为 file_id
        print("错误：CSV中未找到 'file_id' 列。请确保列名正确。")
        return

    # 剔除 file_id 列，只保留数值列用于计算 mean
    # 假设除了 file_id 其他都是 metrics
    metric_columns = [col for col in df.columns if col != 'file_id']
    
    # 将 NaN 填充为 0 (根据需求)
    # 注意：只对数值列填充，避免报错
    df[metric_columns] = df[metric_columns].fillna(0)

    # 用于存储结果的列表
    results = []

    print("数据预处理完成，开始计算分组平均值...")

    # ---------------------------------------------------------
    # 3. 处理 Task 0 (整体任务)
    # 逻辑：file_id 以 '0' 开头
    # ---------------------------------------------------------
    task0_data = df[df['file_id'].str.startswith('0', na=False)]
    
    if not task0_data.empty:
        # 计算平均值
        means = task0_data[metric_columns].mean()
        # 转换为字典并添加标签
        row = means.to_dict()
        row['Group_Label'] = 'Task_0_Overall'
        results.append(row)
        print(f"Task 0 (Overall): 计算完成，包含 {len(task0_data)} 行数据。")
    else:
        print("Task 0 (Overall): 未找到以 '0' 开头的数据。")

    # ---------------------------------------------------------
    # 4. 处理 Task 1-9 (细分任务)
    # 逻辑：前三位 index (Task + Major + Minor)
    # ---------------------------------------------------------
    for task_id in range(1, 10):       # 1 到 9
        for major_id in range(1, 3):   # 1 到 2 (2大类)
            for minor_id in range(1, 5): # 1 到 4 (4小类)
                
                # 构建前缀，例如 "111", "112" ... "924"
                prefix = f"{task_id}{major_id}{minor_id}"
                
                # 筛选数据
                subset = df[df['file_id'].str.startswith(prefix, na=False)]
                
                if not subset.empty:
                    # 计算平均值
                    means = subset[metric_columns].mean()
                    row = means.to_dict()
                    
                    # 格式化标签名称，例如 Task_1_Major_1_Minor_1
                    label_name = f"Task_{task_id}_Major_{major_id}_Minor_{minor_id}"
                    row['Group_Label'] = label_name
                    
                    results.append(row)
                else:
                    # 如果这组没有数据，可以选择跳过
                    pass

    # ---------------------------------------------------------
    # 5. 写入结果
    # ---------------------------------------------------------
    if results:
        # 创建结果 DataFrame
        result_df = pd.DataFrame(results)
        
        # 将 Group_Label 移动到第一列
        cols = ['Group_Label'] + [c for c in result_df.columns if c != 'Group_Label']
        result_df = result_df[cols]
        
        # 保存 CSV
        try:
            result_df.to_csv(output_file, index=False)
            print(f"\n成功！结果已保存至: {output_file}")
            print(f"共生成了 {len(result_df)} 行汇总数据。")
        except Exception as e:
            print(f"写入文件时发生错误: {e}")
    else:
        print("\n没有生成任何结果数据。")

# ==========================================
# 命令行参数解析
# ==========================================
if __name__ == "__main__":
    
    ###     python calculate_metrics.py ./evaluation_result.csv -o my_results.csv

    parser = argparse.ArgumentParser(description="根据file_id前缀计算各任务Metrics的平均值")
    
    # 添加输入文件参数 (位置参数，必选)
    parser.add_argument('input_file', help="输入的CSV文件路径，例如: ./evaluation_result.csv")
    
    # 添加输出文件参数 (可选，带默认值)
    parser.add_argument('-o', '--output', default='calculated_means.csv', 
                        help="输出的CSV文件路径 (默认: calculated_means.csv)")

    args = parser.parse_args()

    # 检查输入文件是否存在
    if os.path.exists(args.input_file):
        calculate_task_means(args.input_file, args.output)
    else:
        print(f"错误: 找不到输入文件 '{args.input_file}'，请检查路径。")