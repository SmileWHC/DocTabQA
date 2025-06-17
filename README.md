DocTabQA: 文档到表格问答系统



一、项目概述



DocTabQA 是一个专注于从长文档生成结构化表格以进行问答的开源项目，提供文档处理、表格生成及评估等完整工具与工作流，助力在问答场景下高效提取信息并结构化呈现。

二、项目结构



### （一）代码目录（`code`）&#xA;



*   `utils`** 子目录**


    *   `doc_text_baseline.py`：实现问答流程中文档转文本操作的基线方法，包含从文档提取相关文本用于答案生成的功能 。


    *   `evaluator.py`：存储评估问答系统性能的代码，涉及计算指标以对比生成的表格答案与真实结果 。


    *   `generate_prompt.py`：负责创建与语言模型交互的提示，引导模型依据文档内容生成表格答案 。


    *   `gpt_doc_table_baseline.py`：定义使用类 GPT 模型从文档生成表格答案的基线方法，搭建模型交互与表格输出格式化流程 。


    *   `gpt_summary_baseline.py`：实现用类 GPT 模型生成文档摘要的基线方法，作为问答前置预处理步骤 。


*   `config.json`：存储项目配置，涵盖模型参数、数据路径等运行时配置项 。


*   `run.py`：项目主入口，编排数据处理、模型运行、结果评估等工作流执行 。


*   `test.py`：包含测试用例，验证项目各组件功能，保障模块正常运作 。


### （二）数据目录（`data`）&#xA;



*   `doc_data.json`：存放项目所用原始文档数据，是生成表格答案的信息源 。


*   `doc_llama_summary.json`：存储 LLaMA 模型生成的文档摘要数据，用于快速信息检索或作为问答输入 。


*   `query_split.json`：包含拆分或预处理后的查询数据，助力复杂问题拆解以适配问答系统 。


*   `structured_doc_sentence.json`：存储句子级结构化文档数据，便于问答时细粒度信息提取 。


*   `table_data.json`：存储用于评估的真实表格数据，与问答系统生成的表格对比 。


### （四）其他文件&#xA;



*   `environment.txt`：罗列项目运行所需依赖与环境设置，用于搭建 Python 环境，如可通过 `pip install -r environment.txt`（依赖格式适配 `pip` 时 ）安装依赖 。


*   `Readme.md`：即当前文件，概述项目、结构及使用说明 。

三、快速开始



### （一）环境准备&#xA;

确保搭建 Python 环境，依据 `environment.txt` 安装依赖，通过 `pip install -r environment.txt`（依赖格式适配 `pip` 时 ）执行。


### （二）数据准备&#xA;

按 `data` 目录现有 JSON 文件结构，将文档数据整理后放入，也可替换为自有数据并保持结构一致 。


### （三）项目运行&#xA;

导航至项目根目录（含 `code` 目录 ），终端执行 `python code/``run.py`，触发数据处理、模型交互、结果生成全流程 。


四、定制与扩展



### （一）模型配置&#xA;

修改 `config.json` 调整模型参数，如变更语言模型输入输出设置、评估指标；新增模型可参照 `utils` 目录现有模型文件（如 `gpt_doc_table_baseline.py` ）创建 Python 文件，更新 `run.py` 纳入工作流 。


### （二）数据处理&#xA;

若数据格式有异，修改 `utils` 目录中数据加载、预处理相关文件；按需更新 `data` 目录 JSON 文件结构适配数据 。


五、评估



执行 `python code/utils/``evaluator.py`（确保数据与生成结果就绪 ），`evaluator.py` 会对比生成的表格答案（或其他输出 ）与参考数据（如 `data` 目录 `table_data.json` ），输出性能指标 。

七、项目创新与架构（补充核心价值说明 ）



DocTabQA 是新颖问答范式，提出从长文档生成表格回答问题任务，构建两阶段框架 DocTabTalk：




*   **检索阶段（AlignLLaMA）**：微调 LLaMA 模型实现查询与文档句子语义对齐，分解问题、用 Sentence - BERT 算相似度获取相关句子，提升检索精度 。


*   **表格生成阶段（TabTalk）**：分阶段生成表格（先定义行列头再填内容 ），借链式思考提示构建层次化表格，支持 HTML 格式输出复杂结构 。
    配套 QTabA 数据集（300 篇金融文档、1.5k 问题 - 表格对 ），在 QTabA、RotoWire 数据集测试，性能优于基线方法 。


    配套 QTabA 数据集（300 篇金融文档、1.5k 问题 - 表格对 ），在 QTabA、RotoWire 数据集测试，性能优于基线方法 。


八、引用



若研究使用 DocTabQA，引用：




```
@article{wang2023doctabqa,


&#x20; title={DocTabQA: Answering Questions from Long Documents Using Tables},


&#x20; author={Wang, Haochen and Hu, Kai and Dong, Haoyu and Gao, Liangcai},


&#x20; journal={arXiv preprint arXiv:2310.00000},


&#x20; year={2023}


}
```
