import copy
import re
import torch
import collections
import string
import json
import time
import numpy as np
import scipy
import evaluate
from tqdm import tqdm
from utils.bart_score import BARTScorer
from utils.generate_prompt import html_evaluate_prompt, markdown_evaluate_prompt
from utils.gpt_api import get_completion
from sacrebleu import sentence_chrf
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import logging

# 配置日志记录
logging.basicConfig(level=logging.ERROR)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


class Evaluator_table(object):
    def __init__(self, args):
        self.args = args
        self.metric = args.evaluate_metric.split('|')

    def table_to_cells(self, table, pred):
        res = []
        for row_idx, row in enumerate(table['rows']):
            for col_idx in range(table["rowname_num"], len(row)):
                header = table['header'][col_idx]
                row_name = ""
                for rn in row[:table["rowname_num"]]:
                    if rn != "":
                        row_name = rn
                content = row[col_idx]
                if pred:
                    res.append((str((header, row_name)), content))
                    res.append((str((row_name, header)), content))
                else:
                    res.append((str((header, row_name)), content))
        return res

    def flat_header(self, rows):
        if rows == []:
            return []
        if isinstance(rows[0], list):
            num_columns = len(rows[0])
        else:
            return rows

        # For each row, copy the content to its empty cells
        for row in rows:
            for col in range(1, num_columns):
                if row[col] == "":
                    row[col] = row[col - 1]

        # Combine all rows and remove bold markdown
        compressed_row = [""] * num_columns
        for col in range(num_columns):
            combined_content = " ".join([row[col].replace("**", "").strip() for row in rows])
            compressed_row[col] = combined_content

        return compressed_row

    def cal_sim_score(self, models, pred, tgt):
        sim_scores, total_len = [], len(tgt)
        for metric in self.metric:
            if metric == "bertscore":
                bertscore_results = models["bertscore"].compute(predictions=pred, references=tgt,
                                                                model_type="model_weight/deberta-xlarge", lang="en")
                sim_scores.append(sum(bertscore_results["f1"]) / total_len)
            elif metric == "rouge":
                rouge_results = models["rouge"].compute(predictions=pred, references=tgt)
                sim_scores.append(rouge_results["rougeL"])
            elif metric == "bleurt":
                bleurt_results = models["bleurt"].compute(predictions=pred, references=tgt)
                sim_scores.append(sum(bleurt_results["scores"]) / total_len)
            elif metric == "bartscore":
                bart_score = models["bartscore"].score(pred, tgt, batch_size=4)
                sim_scores.append(sum(bart_score) / total_len)
            elif metric == "sacrebleu":
                sacrebleu_results = models["sacrebleu"].compute(predictions=pred, references=tgt)
                sim_scores.append(round(sacrebleu_results["score"], 1))
            elif metric == "chrf":
                chrf_results = models["chrf"].compute(predictions=pred, references=tgt)
                sim_scores.append(chrf_results["score"] / 100)
        return sim_scores

    def cal_sim_cell(self, models, preds_all, tgts_all):
        pred_idx_strings = [s for s, c in preds_all]
        pred_idx_embeddings = models["Sentence_Bert"].encode(pred_idx_strings, convert_to_tensor=True)
        target_cells, pred_cells, visited = [], [], set()
        for tgt_str, tgt in tgts_all:
            if tgt == '-' or tgt == "":
                continue
            tgt_idx_embedding = models["Sentence_Bert"].encode(tgt_str, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(tgt_idx_embedding, pred_idx_embeddings).squeeze(0)
            # 对cos_scores进行排序，获取排序后的索引
            sorted_idxs = torch.argsort(cos_scores, descending=True)
            most_similar_idx = sorted_idxs[0].item()
            pred_cells.append(preds_all[most_similar_idx][1])
            target_cells.append(tgt)
        score = self.cal_sim_score(models, pred_cells, target_cells)
        return score

    def structure_check(self, pred_table):
        if isinstance(pred_table["header"][0], list):
            basic_len = len(pred_table["header"][0])
            for row in pred_table["header"]:
                if len(row) != basic_len:
                    return False
        else:
            basic_len = len(pred_table["header"])
        if pred_table["rows"] == []:
            return False
        if isinstance(pred_table["rows"][0], list):
            for row in pred_table["rows"]:
                if len(row) != basic_len:
                    return False
        else:
            if len(pred_table["rows"]) != basic_len:
                return False
        return True

    def calc_score(self, models, pred_table, gold_table):
        try:
            pred_table["header"] = self.flat_header(pred_table["header"])
            gold_table["header"] = self.flat_header(gold_table["header"])
            pred_data = self.table_to_cells(pred_table, True)
            gold_data = self.table_to_cells(gold_table, False)
            cell_score = self.cal_sim_cell(models, pred_data, gold_data)
            return cell_score
        except Exception as e:
            logging.error(f"Error in calc_score: {e}", exc_info=True)
            return 0

    def load_models(self):
        models = {}
        models["Sentence_Bert"] = SentenceTransformer("model_weight/all-mpnet-base-v2")
        for metric in self.metric:
            if metric == "bertscore":
                bertscore = evaluate.load("hf_evaluate/metrics/bertscore")
                models["bertscore"] = bertscore
            elif metric == "rouge":
                rouge = evaluate.load('hf_evaluate/metrics/rouge')
                models["rouge"] = rouge
            elif metric == "bleurt":
                bleurt = evaluate.load("hf_evaluate/metrics/bleurt", module_type="metric")
                models["bleurt"] = bleurt
            elif metric == "bartscore":
                bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
                models["bartscore"] = bart_scorer
            elif metric == "sacrebleu":
                sacrebleu = evaluate.load("hf_evaluate/metrics/sacrebleu")
                models["sacrebleu"] = sacrebleu
            elif metric == "chrf":
                chrf = evaluate.load("hf_evaluate/metrics/chrf")
                models["chrf"] = chrf
        return models

    def calc_scores(self, preds, golds):
        cell_scores, total_len, result = [], len(golds), {}
        models = self.load_models()
        for pred_table, gold_table in tqdm(zip(preds, golds)):
            table_score = self.calc_score(models, pred_table, gold_table)
            cell_scores.append(table_score)
        for i in range(len(self.metric)):
            result[self.metric[i]] = sum(cell_score[i] for cell_score in cell_scores) / total_len
        return result

    def evaluate_table_content(self, preds, golds):
        result = {}
        cell_scores = self.calc_scores(preds, golds)
        result["cell_score"] = cell_scores
        return result

    def evaluate_gpt(self, preds, golds):
        result = {"header_content_score": [], "body_content_score": [], "structure_score": []}
        for pred, gold in tqdm(zip(preds, golds)):
            if self.args.table_type == "html":
                prompt = html_evaluate_prompt(self.args.table_type, pred, gold)
            else:
                prompt = markdown_evaluate_prompt(self.args.table_type, pred, gold)
            retry_count = 0
            while True:
                try:
                    scores = get_completion(prompt, 2, "gpt-35-turbo")
                    score_dict = json.loads(re.search(r'\{[^}]*\}', scores).group())
                    header_content_score, body_content_score, structure_score = float(score_dict["header_content_similarity"]), float(
                        score_dict["body_content_similarity"]), float(score_dict["structural_similarity"])
                    break
                except Exception as e:
                    match = re.search(r"Please retry after (\d+) seconds", str(e))
                    wait_time = int(match.group(1)) if match else 3

                    # 每十次失败后打印一次错误原因
                    if retry_count % 10 == 0:
                        logging.error(f"Failed after {retry_count} attempts: {e}")

                    time.sleep(wait_time)
                    retry_count += 1
            result["header_content_score"].append(header_content_score)
            result["body_content_score"].append(body_content_score)
            result["structure_score"].append(structure_score)
        result["gpt_header_content_score"] = sum(result["header_content_score"]) / len(golds)
        result["gpt_body_content_score"] = sum(result["body_content_score"]) / len(golds)
        result["gpt_structure_score"] = sum(result["structure_score"]) / len(golds)
        return result

    def evaluate_table_structure(self, preds, golds):
        result = {"bert_score": [], "rouge_score": [], "chrf_score": [], "structure_score_1": [], "structure_score_2": []}
        model = SentenceTransformer("model_weight/all-mpnet-base-v2")
        bertscore = evaluate.load("hf_evaluate/metrics/bertscore")
        chrf = evaluate.load("hf_evaluate/metrics/chrf")
        for pred, gold in tqdm(zip(preds, golds)):
            pred_s, pred_depth = [info[0] for info in pred], [info[1] for info in pred]
            pred_emb = model.encode(pred_s, convert_to_tensor=True)
            bert_scores, rouge_scores, chrf_scores, structure_scores_1, structure_scores_2, visited = [], [], [], [], [], set()
            pred_max_depth, tgt_max_depth = 0, 0
            for cell in gold:
                tgt_text, tgt_depth = cell[0], cell[1]
                if tgt_text == "":
                    continue
                tgt_max_depth += tgt_depth
                tgt_emb = model.encode(tgt_text, convert_to_tensor=True)
                cos_scores = util.pytorch_cos_sim(tgt_emb, pred_emb).squeeze(0)
                # 对cos_scores进行排序，获取排序后的索引
                sorted_idxs = torch.argsort(cos_scores, descending=True)
                most_similar_idx = sorted_idxs[0].item()

                pred_dep = pred_depth[most_similar_idx]
                pred_max_depth += pred_dep
                structure_score = abs(pred_dep - tgt_depth)
                pred_text = pred_s[most_similar_idx]
                pred_text, tgt_text = [pred_text], [tgt_text]
                bert_score = bertscore.compute(predictions=pred_text, references=tgt_text,
                                               model_type="model_weight/deberta-xlarge", lang="en")
                chrf_score = chrf.compute(predictions=pred_text, references=tgt_text)["score"] / 100
                bert_scores.append(sum(bert_score["f1"]))
                chrf_scores.append(chrf_score)
                structure_scores_1.append(1 - (structure_score / len(gold)))
                structure_scores_2.append(1 - (structure_score / tgt_max_depth))
            result["bert_score"].append(sum(bert_scores) / len(gold))
            result["chrf_score"].append(sum(chrf_scores) / len(gold))
            result["structure_score_1"].append(sum(structure_scores_1) / len(gold))
            result["structure_score_2"].append(sum(structure_scores_2) / len(gold))
        result["bert_score"] = sum(result["bert_score"]) / len(golds)
        result["chrf_score"] = sum(result["chrf_score"]) / len(golds)
        result["structure_score_1"] = sum(result["structure_score_1"]) / len(golds)
        result["structure_score_2"] = sum(result["structure_score_2"]) / len(golds)
        return result

    def evaluate_result(self, preds, golds):
        result = {}
        if self.args.evaluate_type == 'table_content':
            result["content_result"] = self.evaluate_table_content(preds, golds)
        elif self.args.evaluate_type == 'gpt':
            result["gpt_result"] = self.evaluate_gpt(preds, golds)
        elif self.args.evaluate_type == 'table_structure':
            result["structure_result"] = self.evaluate_table_structure(preds, golds)
        return result