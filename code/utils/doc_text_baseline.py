import re
import os
import json
import time
import torch
import tiktoken
import sys

from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from peft import PeftModel

from tqdm import tqdm
from utils.gpt_api import get_completion, azure_openai
from sentence_transformers import SentenceTransformer, util
from utils.generate_prompt import text_table_prompt, doc_text_prompt, doc_text_query_prompt, llama_split_prompt, llama_summary_prompt
# from utils.Table_Pretraining.tapex.model_interface import TAPEXModelInterface

def generate_table_sentences(markdown_header_table, generate_type="row"):
    table_sentences = []
    table_header, split_find = "", False
    for row in markdown_header_table:
        cells = row.split("|")
        if all(cell.strip() == "---" for cell in cells if cell.strip()):
            split_find = True
            if generate_type == "key_word":
                table_sentences.append(table_header)
            continue
        if split_find:
            if generate_type == "row":
                sentence = table_header + row
            if generate_type == "key_word":
                for cell in cells:
                    if cell.strip() != "":
                        sentence = cell.strip().replace("**", "")
                        break
            table_sentences.append(sentence)
        else:
            table_header += row + "<NEWLINE>"
    return table_sentences

def retrive_table_sentence_mix_1(doc_sentence, query, model, mix_nums):
    doc_embeddings = model.encode(doc_sentence, convert_to_tensor=True)
    output_lists = []
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings).squeeze(0)
    cos_scores = cos_scores.cpu()
    
    for mix in mix_nums:
        cumulative_indices = set()
        threshold, topk = mix[0], mix[1]
        top_k_indices = torch.topk(cos_scores, k=topk).indices.tolist()
        last_indices = []
        for ids in top_k_indices:
            if cos_scores[ids] >= threshold:
                last_indices.append(ids)
        cumulative_indices.update(last_indices)
        matching_indices = sorted(list(cumulative_indices))
        output_lists.append(matching_indices)
    return output_lists

def retrive_table_sentence_mix_2(doc_sentence, query, model, mix_nums):
    doc_embeddings = model.encode(doc_sentence, convert_to_tensor=True)
    output_lists = []
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings).squeeze(0)
    cos_scores = cos_scores.cpu()
    
    for mix in mix_nums:
        cumulative_indices = set()
        threshold, topk = mix[0], mix[1]
        top_k_indices = torch.topk(cos_scores, k=100).indices.tolist()
        last_indices = []
        for ids in top_k_indices:
            if cos_scores[ids] >= threshold and len(last_indices) < topk:
                last_indices.append(ids)
        cumulative_indices.update(last_indices)
        matching_indices = sorted(list(cumulative_indices))
        output_lists.append(matching_indices)
    return output_lists

def retrive_table_sentence(doc_sentence, query, model, threshold=0.5, top_k=10):
    # 此函数代码似乎有问题，未正确返回结果，可根据需求修正
    pass

def sentencebert_doc_text(input_data, model, threshold, top_k, table_type, output_path):
    table_names, doc_sentences, target_table_sentences, querys = input_data
    output = {}
    model = SentenceTransformer(model)
    for i in tqdm(range(len(table_names))):
        all_doc_sentences = doc_sentences[i]["all"]
        num_doc_sentences = doc_sentences[i]["num"]
        all_tgt_table_sentence = target_table_sentences[i]["all"]
        num_tgt_table_sentence = target_table_sentences[i]["num"]
        
        all_predict_table_sentence = retrive_table_sentence(all_doc_sentences, querys[i], model, threshold, top_k)
        num_predict_table_sentence = retrive_table_sentence(num_doc_sentences, querys[i], model, threshold, top_k)
        output[table_names[i]] = {
            "predict_all": all_predict_table_sentence,
            "predict_num": num_predict_table_sentence,
            "target_all": all_tgt_table_sentence,
            "target_num": num_tgt_table_sentence
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
    return output

def calculate_token_num(sentence):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(sentence))
    return num_tokens

def split_batch(doc_sentences):
    max_token_num = 16384
    doc_batchs, batch, token_num, sentence_index = [], [], 0, 0
    for sentence in doc_sentences:
        sentence_token_num = calculate_token_num(sentence)
        if token_num + sentence_token_num > max_token_num:
            doc_batchs.append(batch)
            batch, token_num = [], 0
        batch.append(str(sentence_index) + ": " + sentence)
        sentence_index += 1
        token_num += sentence_token_num
    doc_batchs.append(batch)
    return doc_batchs

def gpt_doc_text(model, input_data, output_path):
    table_names, doc_sentences, target_sentences, querys = input_data
    output = {}
    for i in tqdm(range(len(table_names))):
        all_doc_sentence_batchs = split_batch(doc_sentences[i]["all"])
        num_doc_sentence_batchs = split_batch(doc_sentences[i]["num"])
        query = querys[i]
        all_tgt_table_sentence = target_sentences[i]["all"]
        num_tgt_table_sentence = target_sentences[i]["num"]
        all_predict_table_sentence, num_predict_table_sentence = [], []

        for j in tqdm(range(len(all_doc_sentence_batchs))):
            prompt = doc_text_query_prompt(all_doc_sentence_batchs[j], query)
            all_predict_table_sentence.extend(get_predictions(prompt, model))

        for j in tqdm(range(len(num_doc_sentence_batchs))):
            prompt = doc_text_query_prompt(num_doc_sentence_batchs[j], query)
            num_predict_table_sentence.extend(get_predictions(prompt, model))

        output[table_names[i]] = {
            "predict_all": all_predict_table_sentence,
            "predict_num": num_predict_table_sentence,
            "target_all": all_tgt_table_sentence,
            "target_num": num_tgt_table_sentence
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
    return output

def get_predictions(prompt, model):
    retry_count = 1
    while True:
        try:
            res = get_completion(prompt, 2, model)
            res_list = re.findall(r'\[(.*?)\]', res)
            if res_list:
                res_str = res_list[0]
                res_list = list(res_str.split(','))
            else:
                res_list = []
            return [int(res_element.strip()) for res_element in res_list if res_element.strip().isdigit()]
        except Exception as e:
            match = re.search(r"Please retry after (\d+) seconds", str(e))
            if match:
                wait_time = int(match.group(1))
            else:
                wait_time = 3
            time.sleep(wait_time)

def tapex_doc_text(model, checkpoint, input_data):
    print("run tapex\n")
    model = TAPEXModelInterface(resource_dir="checkpoints_r3_e2",
                                          checkpoint_name="model.pt")
    doc_names, doc_sentences, markdown_header_tables, target_table_sentences = input_data
    output = {}
    for i in tqdm(range(len(doc_names))):
        md_header_table = markdown_header_tables[i]
        tgt_table_sentence = target_table_sentences[i]
        predict_table_sentence = []
        for j in tqdm(range(len(doc_sentences[i]))):
            sentence_label = model.predict(doc_sentences[i][j], md_header_table)
            if int(sentence_label) == 1:
                predict_table_sentence.append(j)
        output[doc_names[i]] = {
            "predict": predict_table_sentence,
            "target": tgt_table_sentence
        }
    return output

def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

def llama_doc_text(model, threshold, top_k, output_path):
    table_data = load_json("dataset/table_data.json")
    target_sentences = get_target_sentences(table_data)
    model = SentenceTransformer(model)
    summary_sentences = load_json("dataset/doc_llama_summary.json")
    doc_sentences = load_json("dataset/doc_data.json")
    sub_querys = load_json("dataset/table_llama_split.json")
    all_sentences = get_all_sentences(doc_sentences, summary_sentences)

    topks = [6]
    thresholds = [0.56]
    mixs = get_mixs(thresholds, topks)
    outputs_topk = {str(mix): {} for mix in mixs}

    for global_id, query_info in tqdm(sub_querys.items()):
        doc_id, table_id = global_id.split("+-")[0], global_id.split("+-")[1]
        sub_query = query_info["sub_query"].split("\n")
        sub_query.append(query_info["query"])
        all_query = sub_query
        all_sentence = all_sentences[doc_id]["sentences"]
        all_index = all_sentences[doc_id]["indexs"]
        all_predict_topk = {}

        for query in all_query:
            predict_sentences = retrive_table_sentence_mix_2(all_sentence, query, model, mixs)
            real_predict = get_real_predict(predict_sentences, all_index)
            all_predict_topk[query] = list(real_predict)

        for i in range(len(mixs)):
            outputs_topk[str(mixs[i])][global_id] = {
                "predict": all_predict_topk,
                "target": target_sentences[global_id]
            }

    with open(output_path, 'w') as f:
        json.dump(outputs_topk, f, indent=4)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_target_sentences(table_data):
    target_sentences = {}
    for table_id, table_info in table_data.items():
        table_name = table_id.split("+-")[1]
        doc_name = table_id.split("+-")[0]
        target = table_info["num_index"]
        target_sentences[table_id] = target
    return target_sentences

def get_all_sentences(doc_sentences, summary_sentences):
    all_sentences = {}
    for doc_id, doc_info in doc_sentences.items():
        all_s, all_index, index = [], [], 0
        summarys = summary_sentences[doc_id]
        for i in range(len(doc_info["num_sentences"])):
            all_s.append(doc_info["num_sentences"][i])
            all_index.append(index)
            cur_summary = summarys[i].split("\n")
            for summary in cur_summary:
                all_s.append(summary)
                all_index.append(index)
            index += 1
        all_sentences[doc_id] = {"sentences": all_s, "indexs": all_index}
    return all_sentences

def get_mixs(thresholds, topks):
    mixs = []
    for threshold in thresholds:
        for tpk in topks:
            mixs.append([threshold, tpk])
    return mixs

def get_real_predict(predict_sentences, all_index):
    real_predict = set()
    for i in range(len(predict_sentences)):
        for predict_index in predict_sentences[i]:
            real_index = all_index[predict_index]
            real_predict.add(real_index)
    return real_predict

def parse_text_into_table(generation_result, num_header_rows=1):
    rows = [[item.replace("none", "").replace(" - ", "").strip().lower() for item in row.split('|')[1:-1]] for row in generation_result]
    header = rows[0: num_header_rows]
    rows = rows[num_header_rows:]
    rows_result = []
    for row in rows:
        if not "---" in " ".join(row) and len(row) > 0:
            rows_result.append(row)
    return {'header': header, 'rows': rows_result}

def doc_text_baseline(args, data_path, output_path):
    llama_doc_text("model_weight/all-mpnet-base-v2", 0.65, 10, "chatgpt/output/final_result/best_recall_query.json")
    # 其他被注释的代码可根据需求恢复使用

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # 可根据需要添加更多参数
    args = parser.parse_args()
    data_path = "path/to/data"  # 可根据实际情况修改
    output_path = "path/to/output"  # 可根据实际情况修改
    doc_text_baseline(args, data_path, output_path)