import os
import json
import time
import torch
import tiktoken
from tqdm import tqdm
from utils.gpt_api import get_completion, azure_openai
from sentence_transformers import SentenceTransformer, util
from utils.generate_prompt import text_table_prompt, summary_question_prompt, summary_refine_prompt

# 配置管理类
class Config:
    def __init__(self, args):
        self.gpt_model = args.gpt_model
        self.sentencebert_model = args.sentencebert_model
        self.table_type = args.table_type
        self.topk = args.topk
        self.data_path = "data/structured_doc_sentence.json"
        self.output_path = "chatgpt/output/text_to_table/summary/"
        self.doc_segment_output_path = os.path.join(self.output_path, f"doc_summary/doc_segment/{self.sentencebert_model}_{self.topk}.json")
        self.segment_summary_output_path = os.path.join(self.output_path, f"doc_summary/segment_summary/{self.gpt_model}_{self.table_type}_{self.sentencebert_model}_{self.topk}.json")
        self.summary_table_output_path = os.path.join(self.output_path, f"doc_summary/summary_table/{self.gpt_model}_{self.sentencebert_model}_{self.topk}.json")
        self.doc_table_output_path = os.path.join(self.output_path, f"doc_summary/doc_table/{self.gpt_model}_{self.sentencebert_model}_{self.topk}.json")

def calculate_token_num(sentence):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(sentence))
    return num_tokens

def segment_text(doc_sentences, token_limit = 2048):
    segement_sentences, segmentation, token_num = [], "", 0
    for sentence in doc_sentences:
        sentence_token_num = calculate_token_num(sentence)
        if token_num + sentence_token_num > token_limit:
            segement_sentences.append(segmentation)
            segmentation = ""
            token_num = 0
        segmentation += sentence + " "
        token_num += sentence_token_num
    segement_sentences.append(segmentation)
    return segement_sentences

def retrieve_text(segmentations, sentence, model_name, top_k):
    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(sentence, convert_to_tensor=True)
    segmentation_embeddings = model.encode(segmentations, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(sentence_embeddings, segmentation_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    top_segmentations = [segmentations[idx] for idx in top_results.indices]
    return top_segmentations

def summary_text(segmentations, sentence, model):
    summary = ""
    for i in range(len(segmentations)):
        if i == 0:
            prompt = summary_question_prompt(segmentations[i], sentence)
        else:
            prompt = summary_refine_prompt(segmentations[i], sentence, summary)
        retry_count = 0
        while retry_count < 100:
            try:
                res = get_completion(prompt, 0, model)
                break
            except:
                retry_count += 1
                time.sleep(3)
                continue
        time.sleep(5)
        summary = res
    return summary

def generate_table(summaries, table_header, model):
    prompt = text_table_prompt(summaries, table_header)
    retry_count = 0
    while retry_count < 100:
        try:
            res = get_completion(prompt, 0, model)
            break
        except:
            retry_count += 1
            time.sleep(3)
            continue
    time.sleep(5)
    return res

def generate_table_sentences(markdown_header_table):
    table_sentences = []
    table_header, split_find = "", False
    for row in markdown_header_table:
        cells = row.split("|")
        if all(cell.strip() == "---" for cell in cells if cell.strip()):
            split_find = True
            continue
        if split_find:
            sentence = table_header + row
            table_sentences.append(sentence)
        else:
            table_header += row + "<NEWLINE>"
    return table_sentences

def gpt_summary_table(doc_sentences, table_header, sentence_model, gpt_model, sentence_topk):
    table_sentences = generate_table_sentences(table_header)
    if gpt_model == "gpt-35-turbo":
        token_limit = 2048
    else:
        token_limit = 16384
    segmentations = segment_text(doc_sentences, token_limit)
    if len(segmentations) < sentence_topk:
        sentence_topk = len(segmentations)
    segments, summaries, segment_summary = [], [], []
    for table_sentence in table_sentences:
        summary = summary_text(segmentations, table_sentence, gpt_model)
        summaries.append(summary)
        segment_summary.append({"table": table_sentence, "summary": summary})
    predict_table = generate_table(summaries, table_header, gpt_model)
    print(summaries)
    return predict_table, segments, summaries, segment_summary

def process_data(config):
    data = json.load(open(config.data_path))
    output_summary_table, output_doc_segment, output_segment_summary, output_doc_table = {}, {}, {}, {}
    for doc_id, doc_info in tqdm(data.items(), position=0, leave=True):
        for table_id, table_info in tqdm(doc_info['table_info'].items(), position=1, leave=True):
            predict_table, predict_segments, predict_summaries, predict_seg_sum = gpt_summary_table(
                doc_info['doc_sentences'], table_info['markdown_header_subtable'],
                config.sentencebert_model, config.gpt_model, config.topk)
            target_table = table_info['markdown_subtable']
            target_segments = table_info['sentences']
            output_doc_table[doc_id + '_' + table_id] = {
                "predict": predict_table,
                "target": target_table
            }
            output_doc_segment[doc_id + '_' + table_id] = {
                "predict": predict_segments,
                "target": target_segments
            }
            output_segment_summary[doc_id + '_' + table_id] = {
                "predict": predict_seg_sum
            }
            output_summary_table[doc_id + '_' + table_id] = {
                "summary": predict_summaries,
                "table": target_table
            }
    # 保存输出文件
    save_output(output_doc_table, config.doc_table_output_path)
    save_output(output_doc_segment, config.doc_segment_output_path)
    save_output(output_segment_summary, config.segment_summary_output_path)
    save_output(output_summary_table, config.summary_table_output_path)

def save_output(data, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def gpt_summary_baseline(args):
    config = Config(args)
    process_data(config)

# 示例调用
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_model", default="gpt-35-turbo")
    parser.add_argument("--sentencebert_model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--table_type", default="default")
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()
    gpt_summary_baseline(args)