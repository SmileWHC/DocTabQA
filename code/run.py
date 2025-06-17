import re
import os
import json
import time
import torch
import argparse
import tiktoken
from tqdm import tqdm
from utils.text_table_baseline import text_table_baseline
from utils.gpt_doc_table_baseline import gpt_doc_table_baseline, pipeline_baseline
from utils.gpt_summary_baseline import gpt_summary_baseline
from utils.doc_text_baseline import doc_text_baseline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载配置文件
def load_config():
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # 默认配置
        return {
            "model_types": ["gpt", "sentence_bert", "tapex", "llama"],
            "gpt_models": ["gpt-4", "gpt-4-32k", "gpt-35-turbo"],
            "table_types": ["markdown", "keyword", "query"],
            "bert_models": ["model_weight/all-MiniLM-L6-v2", "model_weight/all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "multi-qa-distilbert-cos-v1"],
            "data_path": "data/final_data_1.json",
            "output_path": "output/final_result/",
            "threshold": 0.5,
            "top_k": 10,
            "run_types": ["text_table", "doc_text", "summary", "doc_table"]
        }

# 文本到表格参数配置
def text_table_args(config):
    config["model"] = config["gpt_models"][1]
    config["data_path"] = "output/final_result/best_recall_query.json"
    config["output_path"] = "output/final_result/best_recall_query.json"
    return config

# 文档到文本参数配置
def doc_text_args(config):
    model_type = config["model_types"][1]
    config["table_type"] = config["table_types"][2]
    if model_type == "gpt":
        config["model"] = config["gpt_models"][1]
        config["output_path"] += "{}_recall.json".format(config["model"])
    elif model_type == "tapex":
        config["model"] = "tapex.large"
        config["checkpoint"] = "checkpoints_r3_e2/model.pt"
        config["output_path"] += "doc_to_text/tapex/tapex_r3_e2.json"
    elif model_type == "llama":
        config["model"] = "llama"
        config["output_path"] += "doc_to_text/query/llama/{}_final.json".format(config["model"])
    else:
        config["model"] = config["bert_models"][0]
        config["top_k"] = 60
        config["threshold"] = 0.8
        config["output_path"] += "MiniLM_{}_recall.json".format(config["threshold"])
    return config

# 摘要参数配置
def summary_args(config):
    config["gpt_model"] = config["gpt_models"][0]
    config["sentencebert_model"] = config["bert_models"][0]
    config["table_type"] = config["table_types"][0]
    config["top_k"] = config["top_k"]
    config["output_path"] = config["output_path"]
    return config

# 文档到表格参数配置
def doc_table_args(config):
    config["model"] = config["gpt_models"][1]
    config["output_path"] += "doc_table/{}_query_sb_v1.json".format(config["model"])
    return config

# 运行不同的基线模型
def run(run_type, config):
    if run_type == "text_table":
        config = text_table_args(config)
        text_table_baseline(config, config["data_path"], config["output_path"])
    elif run_type == "doc_text":
        config = doc_text_args(config)
        doc_text_baseline(config, config["data_path"], config["output_path"])
    elif run_type == "summary":
        config = summary_args(config)
        gpt_summary_baseline(config, config["data_path"], config["output_path"])
    elif run_type == "doc_table":
        config = doc_table_args(config)
        pipeline_baseline(config["data_path"], config["output_path"])
        # gpt_doc_table_baseline(config, config["data_path"], config["output_path"])

if __name__ == "__main__":
    run_baseline = ["summary", "doc_table", "doc_text", "text_table"]
    config = load_config()
    run_type = run_baseline[3]
    run(run_type, config)