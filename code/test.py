import json
import time
import tiktoken
import random
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
# from gpt_api import get_completion, azure_openai
# from generate_prompt import text_table_prompt
# from sentence_transformers import SentenceTransformer, util

# 工具函数
def calculate_token_num(sentence):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(sentence))
    return num_tokens

def fix_index_false(label, data):
    for label_id, label_info in label.items():
        last_underscore_index = label_id.rfind('_')
        doc_id = label_id[:last_underscore_index]
        table_id = label_id[last_underscore_index + 1:]
        label_info['target'] = data[doc_id]['table_info'][table_id]['sentences_index']
    return label

def check_line(symbol, line):
    flag = True
    for cell in line.split('|')[1:-1]:
        if cell.strip() == '':
            continue
        if symbol not in cell.strip():
            flag = False
            break
    return flag

# 主要功能函数
def gpt_text_table(model, data_path, output_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    table_names, table_sentences, markdown_header_tables, markdown_target_tables = [], [], [], []
    for doc_id, doc_info in data.items():
        for table_id, table_info in doc_info['table_info'].items():
            table_names.append(doc_id + '_' + table_id)
            table_sentences.append(table_info['sentences'])
            markdown_header_tables.append(table_info['markdown_header_subtable'])
            markdown_target_tables.append(table_info['markdown_subtable'])
    output = {}
    for i in tqdm(range(len(table_names))):
        prompt = text_table_prompt(table_sentences[i], markdown_header_tables[i])
        retry_count = 0
        while retry_count < 8:
            try:
                # res = azure_openai(model, prompt)
                res = get_completion(prompt, 0, model)
                output[table_names[i]] = {
                    "predict": res,
                    "target": markdown_target_tables[i]
                }
                break
            except:
                retry_count += 1
                time.sleep(3)
                continue
        time.sleep(5)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

def correct_table_order(table_data):
    last_header_index = -1
    separator_index = -1
    cell_num = max(line.count('|') - 1 for line in table_data)

    # Iterate over the lines to find the last full header index and the separator index
    for index, line in enumerate(table_data):
        if line == '':
            continue
        if check_line("**", line):  # Check if the line is a full header line
            last_header_index = index
        elif check_line("---", line):  # Find the first separator line
            separator_index = index
    # Determine the correct position to insert the separator line
    insert_position = last_header_index + 1
    separator_line = '| ' + ' | '.join(['---'] * cell_num) + ' |'
    if separator_index != -1:
        # If the separator is not right after the last header, move the separator to the correct position
        if separator_index != last_header_index + 1:
            # Remove the existing separator line
            table_data.pop(separator_index)
            # Adjust the insert position if the separator was before the last header line
            if separator_index < last_header_index:
                insert_position -= 1
            # Insert the separator line at the correct position
            table_data.insert(insert_position, separator_line)
    else:
        # Insert the separator line at the correct position
        table_data.insert(insert_position, separator_line)
    return table_data

def eval_sentence_bert():
    with open("chatgpt/output/doc_to_text/sentence_bert.json", 'r') as f:
        data = json.load(f)
    # Create the DataFrame
    df = pd.DataFrame(data)

    # Sort the DataFrame by model, threshold, and topk in increasing order
    df.sort_values(by=['model', 'threshold', 'topk'], inplace=True)

    # Round the precision, recall, and f1 to three decimal places
    df['precision'] = df['precision'].round(3)
    df['recall'] = df['recall'].round(3)
    df['f1'] = df['f1'].round(3)

    # Convert the DataFrame to a markdown table string
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)

def eval_doc_segment():
    with open("chatgpt/output/doc_summary/doc_segment/all-mpnet-base-v2_3.json", "r") as f:
        data = json.load(f)
    recall = []
    for doc_id, doc_info in data.items():
        target_num = len(doc_info['target'])
        predict_num = 0
        for sentence in doc_info['target']:
            flag = False
            for segment in doc_info['predict']:
                for segment_sentence in segment:
                    if sentence in segment_sentence:
                        flag = True
                        predict_num += 1
                        break
                if flag:
                    break
        recall.append(predict_num / target_num)
    print(sum(recall) / len(recall))

def cal_table_sentence_header_similarity():
    model = SentenceTransformer("all-mpnet-base-v2")
    with open("data/structured_data_all_docsentence_v1.json", "r") as f:
        data = json.load(f)
    cos_scores = []
    for doc_id, doc_info in tqdm(data.items()):
        for table_id, table_info in doc_info["table_info"].items():
            row_to_sentences = {}
            for entry in table_info["sub_table"]:
                # 获取当前条目的行标识
                row_id = entry['matrix_subtable_row_id']

                # 如果当前行标识还没有在字典中，则添加一个新的条目
                if row_id not in row_to_sentences:
                    row_to_sentences[row_id] = {'row_name': None, 'sentence_indices': []}

                # 如果条目是header，则更新行名
                if entry['is_header']:
                    row_to_sentences[row_id]['row_name'] = entry['value']

                # 如果有相关的句子索引，则添加到句子索引列表中
                if entry['sentence_id'] != -1:
                    row_to_sentences[row_id]['sentence_indices'].extend(entry['sentence_id'])
        for row_id, row_info in row_to_sentences.items():
            row_name = row_info['row_name']
            sentence_indices = row_info['sentence_indices']
            for sentence_index in sentence_indices:
                sentence = table_info['sentences'][sentence_index]
                # 计算当前行名和句子的相似度
                cos_scores.append(util.pytorch_cos_sim(model.encode(row_name), model.encode(sentence)))
    print("cos_score < 0.3: {}, > 0.3: {}".format(len([score for score in cos_scores if score < 0.3]), len([score for score in cos_scores if score > 0.3])))
    print("min: {}, max: {}, mean: {}".format(min(cos_scores), max(cos_scores), sum(cos_scores) / len(cos_scores)))

def parse_text_into_table_v1(generation_result, num_header_rows=1):
    generation_result = "|" + "|".join(generation_result.split("|")[1:-1]) + "|"
    if "notes" in generation_result:
        rows = [[item.replace("none", "").replace(" - ", "").strip().lower() for item in row.split('|')[1:-2]] for row in generation_result.strip().split('\n')]
    else:
        rows = [[item.replace("none", "").replace(" - ", "").strip().lower() for item in row.split('|')[1:-1]] for row in generation_result.strip().split('\n')]
    header = rows[0]
    rows = rows[num_header_rows:]  # for evluating cot result
    rows_result = []
    for row in rows:
        if not "---" in " ".join(row) and len(row) > 0:
            rows_result.append(row)
    return {'header': header, 'rows': rows_result}

def parse_text_into_table(generation_result):
    generation_result = generation_result.replace("<NEWLINE>", "\n")
    rows = [[item.replace("none", "-").strip().lower() for item in row.split('|')[1:-1]] for row in generation_result.strip().split('\n')]
    header, rows_result, split_row = [], [], False
    for row in rows:
        if all(cell.strip() == "---" for cell in row):
            split_row = True
        else:
            if split_row:
                rows_result.append(row)
            else:
                header.append(row)
    return {'header': header, 'rows': rows_result}

def parse_html_into_table(html_code):
    soup = BeautifulSoup(html_code, 'html.parser')
    table = soup.find('table')
    header_rows = soup.find('thead').find_all('tr')
    output = {'header': [], 'rows': []}
    for row in header_rows:
        row = row.find_all('th')
        row_cells = []
        for cell in row:
            row_cells.append(cell.text.strip().lower())
            if "colspan" in cell.attrs:
                for i in range(int(cell.attrs["colspan"]) - 1):
                    row_cells.append("")
        output['header'].append(row_cells)
    body_rows = soup.find('tbody').find_all('tr')
    for row in body_rows:
        row_name = row.find('th').text.strip().lower()
        row_cells = [row_name]
        for cell in row.find_all('td'):
            row_cells.append(cell.text.strip().lower())
        output["rows"].append(row_cells)

    return output

def add_hierarchy_row():
    with open("dataset/table_data.json", "r") as f:
        data = json.load(f)
    for doc_id, doc_info in tqdm(data.items()):
        if doc_info["col_hierachy"] == 1:
            table = []
            doc_info["table"] = table
    with open("dataset/table_data_rowh.json", "w") as f:
        json.dump(data, f, indent=4)

def check_data():
    with open("dataset/table_data_rowh.json", "r") as f:
        data = json.load(f)
    for doc_id, doc_info in data.items():
        def markdown_check(table):
            row_num = table[0].count("|")
            for row in table:
                if row.count("|") != row_num:
                    return False
            return True

        if not markdown_check(doc_info["table"]):
            print(doc_id)

        def html_check(html):
            html_string = "\n".join(html)
            th_open_count = html_string.count("<th") - 1
            th_close_count = html_string.count("</th>")
            tr_open_count = html_string.count("<tr>")
            tr_close_count = html_string.count("</tr>")
            td_open_count = html_string.count("<td>")
            td_close_count = html_string.count("</td>")
            if th_open_count != th_close_count:
                print("th", doc_id)
            if tr_open_count != tr_close_count:
                print("tr", doc_id)
            if td_open_count != td_close_count:
                print("td", doc_id)
            if th_open_count != th_close_count or tr_open_count != tr_close_count or td_open_count != td_close_count:
                return False
            return True

        if not html_check(doc_info["html"]):
            print(doc_id)
        html_cells = [cell.strip() for cell in doc_info["html"]]
        doc_info["html_code"] = "\n".join(html_cells)
    with open("dataset/table_data_final.json", "w") as f:
        json.dump(data, f, indent=4)

def count_data():
    with open("dataset/table_data_final.json", "r") as f:
        data = json.load(f)
    all_table_tokens, all_row_num, all_col_num, all_cell_num = [], [], [], []
    for doc_id, doc_info in data.items():
        col_num = doc_info["table"][0].count("|") - 1
        row_num = len(doc_info["table"])
        cell_num = col_num * row_num
        table = "\n".join(doc_info["table"])
        table_tokens = calculate_token_num(table)
        all_table_tokens.append(table_tokens)
        all_row_num.append(row_num)
        all_col_num.append(col_num)
        all_cell_num.append(cell_num)
    print("table_tokens: {}, row_num: {}, col_num: {}, cell_num: {}".format(sum(all_table_tokens) / len(all_table_tokens),
                                                                            sum(all_row_num) / len(all_row_num),
                                                                            sum(all_col_num) / len(all_col_num),
                                                                            sum(all_cell_num) / len(all_cell_num)))
    with open("dataset/doc_data.json", "r") as f:
        data = json.load(f)
    all_doc_tokens = []
    for doc_id, doc_info in data.items():
        all_doc_sentence = ""
        for sentence in doc_info["all_sentences"]:
            all_doc_sentence += sentence + " "
        doc_tokens = calculate_token_num(all_doc_sentence)
        all_doc_tokens.append(doc_tokens)
    print("doc_tokens: {}".format(sum(all_doc_tokens) / len(all_doc_tokens)))

def e2e_count_data():
    with open("text_to_table/data/rotowire/train_rotowire.json", "r") as f:
        data = json.load(f)
    all_table_tokens, all_row_num, all_col_num, all_cell_num, doc_tokens = [], [], [], [], []

    for row in tqdm(data):
        table = row["output"]
        table_rows = table.split("<NEWLINE>")
        col_num = table_rows[0].count("|") - 1
        row_num = len(table_rows)
        sentence_tokens = calculate_token_num(row["input"])
        table_tokens = calculate_token_num(table)
        all_table_tokens.append(table_tokens)
        all_row_num.append(row_num)
        all_col_num.append(col_num)
        all_cell_num.append(col_num * row_num)
        doc_tokens.append(sentence_tokens)
    print("table_tokens: {}, row_num: {}, col_num: {}, cell_num: {}, doc_tokens: {}".format(sum(all_table_tokens) / len(all_table_tokens),
                                                                                             sum(all_row_num) / len(all_row_num),
                                                                                             sum(all_col_num) / len(all_col_num),
                                                                                             sum(all_cell_num) / len(all_cell_num),
                                                                                             sum(doc_tokens) / len(doc_tokens)))

def evaluate_gpt_recall():
    with open("chatgpt/output/final_result/gpt-4-32k_recall.json", "r") as f:
        data = json.load(f)
    topk_recall = [10, 20, 30, 40, 50, 60]
    topk_recall_score = [[], [], [], [], [], []]
    for i in range(20):
        for doc_id, recall_info in data.items():
            target = recall_info["target_num"]
            predict = recall_info["predict_num"]
            for i in range(len(topk_recall)):
                topk = topk_recall[i]
                if topk > len(predict):
                    topk = len(predict)
                predict_r = random.sample(predict, topk)
                recall_num = len(set(predict_r) & set(target))
                recall_score = recall_num / len(target)
                topk_recall_score[i].append(recall_score)
    for i in range(len(topk_recall)):
        print("topk: {}, recall: {}".format(topk_recall[i], sum(topk_recall_score[i]) / len(topk_recall_score[i]) * 100))

# 主程序入口
if __name__ == "__main__":
    # 可以在这里调用需要执行的函数
    e2e_count_data()