import os
import re
import json
import time
import torch
import tiktoken
from tqdm import tqdm
from utils.gpt_api import get_completion, azure_openai
from utils.generate_prompt import text_table_prompt, summary_question_prompt, summary_refine_prompt, doc_table_prompt, query_table_prompt, query_table_updated_prompt, ttt_prompt, ttt_tabtalk_prompt

# 配置文件路径
CONFIG = {
    "table_data": "dataset/table_data_final.json",
    "doc_data": "dataset/doc_data.json",
    "rotowire_data": "dataset/test_rotowire.json",
    "pipeline_output": "output/pipeline_final.json",
    "rotowire_simple_output": "output/rotowire_simple_final.json",
    "rotowire_tabtalk_output": "output/rotowire_tabtalk_final.json"
}

def calculate_token_num(sentence):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(sentence))
    return num_tokens

def segment_text(doc_sentences, token_limit = 16384):
    segement_sentences, segmentation, token_num, sentence_index = [], "", 0, 1
    for sentence in doc_sentences:
        sentence_token_num = calculate_token_num(sentence)
        if token_num + sentence_token_num > token_limit:
            segement_sentences.append(segmentation)
            segmentation = ""
            token_num = 0
            sentence_index = 1
        segmentation += str(sentence_index) + ". " + sentence + "\n"
        sentence_index += 1
        token_num += sentence_token_num
    segement_sentences.append(segmentation)
    return segement_sentences

def segment_prompt(query, doc_sentences):
    token_num, sentence_index = 0, 1
    t_input = "Query:\n{}\nDocument sentences:\n".format(query)
    for sentence in doc_sentences:
        sentence_token_num = calculate_token_num(sentence)
        if token_num + sentence_token_num > 30000:
            break
        t_input += str(sentence_index) + ". " + sentence + "\n"
        sentence_index += 1
        token_num += sentence_token_num
    prompt = query_table_prompt(t_input)
    return prompt

def gpt_text_table(model, query, sentences):
    prompt = query_table_prompt(query, sentences)
    return _get_completion_with_retry(prompt, model)

def gpt_text_table_updated(model, table, query, sentences):
    prompt = query_table_updated_prompt(table, query, sentences)
    return _get_completion_with_retry(prompt, model)

def _get_completion_with_retry(prompt, model):
    retry_count = 0
    while True:
        try:
            res = get_completion(prompt, 2, model)
            print(res)
            return res
        except Exception as e:
            match = re.search(r"Please retry after (\d+) seconds", str(e))
            wait_time = int(match.group(1)) if match else 3

            # 每十次失败后打印一次错误原因
            if retry_count % 10 == 0:
                print(f"Failed after {retry_count} attempts: {e}")
            retry_count += 1

            time.sleep(wait_time)

def gpt_doc_table_baseline(args, data_path, output_path):
    data, model = json.load(open(data_path)), args.model
    output = {}
    if model == "gpt-35-turbo":
        token_limit = 2048
    else:
        token_limit = 8192
    for doc_id, doc_info in tqdm(data.items()):
        doc_sentences = doc_info['doc_sentences']
        segmentations = segment_text(doc_sentences, token_limit)
        for table_id, table_info in doc_info['table_info'].items():
            predict_table = table_info['markdown_header_subtable']
            for i in tqdm(range(len(segmentations))):
                predict_table = gpt_text_table(model, segmentations[i], predict_table)
            output[doc_id + "_" + table_id] = {
                "predict": predict_table,
                "target": table_info['markdown_subtable']
            }
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=4)

def pipeline_baseline():
    table_data = json.load(open(CONFIG["table_data"]))
    doc_data = json.load(open(CONFIG["doc_data"]))
    model = "gpt-4-32k"
    output, token_limit = {}, 16384
    for global_id, table_info in tqdm(table_data.items()):
        doc_id, table_id = global_id.split("+-")[0], global_id.split("+-")[1]
        t_output = table_info["html_code"]
        doc_sentences = doc_data[doc_id]["num_sentences"]
        query = table_info["query"]
        prompt = segment_prompt(query, doc_sentences)
        ans = _get_completion_with_retry(prompt, model)
        output[global_id] = {
            "predict": ans,
            "target_html": t_output,
        }
        with open(CONFIG["pipeline_output"], 'w') as f:
            json.dump(output, f, indent=4)

def ttt_baseline():
    data = json.load(open(CONFIG["rotowire_data"]))
    index = 0
    output = {}
    for row in tqdm(data):
        t_input = row["input"]
        t_output = row["output"][row["output"].find("Player") + 18:]
        prompt = ttt_prompt(t_input)
        ans = _get_completion_with_retry(prompt, "gpt-4-32k")
        output[index] = {
            "predict": ans,
            "target_markdown": t_output,
        }
        with open(CONFIG["rotowire_simple_output"], 'w') as f:
            json.dump(output, f, indent=4)
        index += 1

def ttt_baseline_tabtalk():
    data = json.load(open(CONFIG["rotowire_data"]))
    index = 0
    output = {}
    for row in tqdm(data):
        t_input = row["input"]
        t_output = row["output"][row["output"].find("Player") + 18:]
        prompt = ttt_tabtalk_prompt(t_input)
        ans = _get_completion_with_retry(prompt, "gpt-4-32k")
        output[index] = {
            "predict": ans,
            "target_markdown": t_output,
        }
        with open(CONFIG["rotowire_tabtalk_output"], 'w') as f:
            json.dump(output, f, indent=4)
        index += 1

if __name__ == "__main__":
    pipeline_baseline()
    print("Pipeline finished.")
    ttt_baseline()
    print("TTT finished.")
    ttt_baseline_tabtalk()
    print("TTT Tabtalk finished.")