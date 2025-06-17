import os
import json

from tqdm import tqdm

def text_table_prompt(sentences, table_header):
    instruction = "Role: As an AI expert in financial data analysis, you are tasked to generate complete financial tables based on the provided table headers and accompanying guidelines.\nInput Content: You are supplied with table headers in Markdown format and accompanying guidelines which are the sentences related to the contents of the table.\n"
    table_structure = "<NEWLINE>".join(table_header)
    candidate_sentences = "\n".join(sentences)
    input = "Table Header: " + table_structure + "\n" + "Guidelines: " + candidate_sentences + "\n"
    output = "Output Format Requirements: Your output should be a Markdown-formatted, complete table and do not alter the table structure. Some cells may remain empty based on the guidelines. Use <NEWLINE> to represent line breaks within the output."
    prompt = instruction + input + output
    
    return prompt

def doc_text_prompt(sentences, table_header):
    instruction = "Role task: As an AI expert in data analysis, given all the sentences of a document and the header and structure of a table, I'd like you to help me sift through all the sentences to find some useful ones that predict the cell contents.\nOutput Considerations: Returns only one list and nothing else. each element of the list is the number of the filtered sentence.If no useful sentences were found in the given document, return an empty list.\nInput Content: Analyze the structure of the following table, including its column names, and row names and choose useful sentences from document:\n"
    input_table = "Table: " + "<NEWLINE>".join(table_header) + "\n"
    input_doc = "Document Sentences: " + "\n".join(sentences) + "\n"
    output = "Output:Returns only one list and nothing else. Please select useful sentences and give them their serial numbers to output in the form of a list. If no useful sentences were found in the given document, return an empty list."
    prompt = instruction + input_table + input_doc + output

    return prompt

def doc_text_query_prompt(sentences, query):
    prompt = """
>>>>>>Role task: 
As an AI expert in data analysis, given all the sentences of a document and a query, I'd like you to help me sift through all the sentences to pick out some sentences that are relevant to the query.
>>>>>>Output Considerations:
Returns only one list and nothing else. each element of the list is the number of the filtered sentence.If no useful sentences were found in the given document, return an empty list.
>>>>>>For example:
Query:
The value of sales to MiTAC Holdings and its affiliates during fiscal years ended November 30, 2017.
Document Sentences:
0. The Company\u2019s sales to MiTAC Holdings and its affiliates during fiscal years ended November 30, 2017, 2016, and 2015 totaled $1,202, $1,809 and $1,290, respectively
1. Today is a good day
2. The Company\u2019s sales to MiTAC Holdings and its affiliates during fiscal years ended November 30, 2017 is $1,202. 
Output:
[0, 2]
>>>>>>Input Content:
Query:
{}
Document Sentences:
{}
>>>>>>Output:
    """.format(query, "\n".join(sentences))
    return prompt

def summary_question_prompt(segmentation, sentence):
    prompt = """
>>>>>Your Task:
Given a segment of a financial report and table header.
You need to summarize the information related the table header. 
>>>>>Example: 
Financial report 's segment: For the company A in 2022Q3 , revenue is $1 .2345 billion; the net income is $50.1245 million. The board members were very satisfied, because they thought company has made great strides this quarter. 
----
Table: 
| Company A | <NEWLINE> | 2022 | 2023 | <NEWLINE> | Net income | - | <NEWLINE> | Revenue | - | <NEWLINE>
----
Summary: 
For company A at 2022Q3 , net income is $50 .125 million , revenue is $1 ,234.500 million. 
>>>>>Question: 
Financial report 's segment: {} 
----
Table headers: {} 
----
>>>>>Output: 
Summary:
""".format(segmentation, sentence)
    return prompt

def summary_refine_prompt(segmentation, sentence, pre_summary):
    prompt = """
>>>>>Your Task: 
Given a segment of a financial report , a summary of the previous segments and table header. 
Your should combine the information related to the table to generate a new summary.
>>>>>Example: 
Financial report 's segment: In the fourth quarter, while progress was made, it was not as dramatic as expected. For the company A in 2022Q4, the net income is $5 billion. 
----
Old summary: For the company A, the net income in 2022Q1 is $3.125 million; the net income in 2022Q2 is $123.000 million; the net income in 2022Q3 is $0.123 million. 
----
Table headers : | Company A | <NEWLINE> | 2022Q1| 2022Q2 | 2022Q3 | 2022Q4 | <NEWLINE> | Net income | - | <NEWLINE> 
----
New summary: The net income of 2022Q1, 2022Q2, 2022Q3 and 2022Q4 is $3.125 million, $123.000 million, $0.123 million, $5 billion respectively
>>>>>Question:
Financial report 's segment: {} 
----
Old summary: {} 
----
Table headers: {}
----
>>>>>Output: 
New summary:
""".format(segmentation, pre_summary, sentence)
    return prompt
    

def doc_table_prompt(segment, table):
    prompt = """
>>>>>Your Task:
Given a text segment of a financial report and an incomplete table.
You need to supplement the table according to the text, do not change the table structure, just fill the cells.
Returns the table as markdown, with <NEWLINE> as the newline character for the table.
>>>>>Example: 
Financial report 's segment: 
For the company A in 2022Q3 , revenue is $1 .2345 billion; the net income is $50.1245 million. The board members were very satisfied, because they thought company has made great strides this quarter. 
----
Table: 
| Company A | - |<NEWLINE> | 2022 | 2023 | <NEWLINE> | --- | --- | <NEWLINE> | Net income | - | - |<NEWLINE> | Revenue | - | - | <NEWLINE>
----
Updated Table:
| Company A | - |<NEWLINE> | 2022 | 2023 | <NEWLINE> | --- | --- | <NEWLINE> | Net income | $50.1245 million | - | <NEWLINE> | Revenue | $1.2345 billion | - | <NEWLINE>
----
>>>>>Input: 
Financial report 's segment: 
{} 
----
Table: 
{} 
----
>>>>>Output: 
Updated Table:
""".format(segment, table)
    return prompt

def label_box_query_prompt(table):
    prompt = """
>>>>>Your Role:
Your role as an AI data annotation expert is to create queries that dictate the structure and content of data tables. 
>>>>>Your Task:
These queries are key in extracting relevant sentences from documents, with the structure of the tables being defined by the query itself. Your main objective is to formulate precise queries that serve both as tools for data extraction and as blueprints for the table's layout, focusing on the essential elements needed for the table. The challenge is to ensure these queries are comprehensive yet specific, guiding the creation of well-structured and informative tables.The query should be in the form of a question or a requirement
>>>>>Input:
Table:
{}
>>>>>Output (Query):
""".format(table)
    return prompt
# data_path = "/home/TableSense/hdd10T/whc/TextToTable/icdar_ttt/data/structured_data_v1.json"

def split_query_prompt(query, table):
    prompt = """
>>>>>Your Role:
You will be responsible for splitting a query into multiple sub-queries, which can be referenced in a table strongly related to the query.
>>>>>Your Task:
Given a query and a table that corresponds to it in its entirety, you are now expected to split that query into queries that correspond to individual cells.
>>>>>For example:
Query: What are the revenues, cost of revenues, gross profit, and interest expense for the three and nine months ended September 30, 2021?
Table:
|   | **Three Months Ended September 30, ** | **Nine Months Ended September 30, ** |
|   | **2021 ** | **2021 ** |
| --- | --- | --- |
| **Revenues ** | $ 1  | $ 1  |
| **Cost of revenues ** | 83  | 83  |
| **Gross profit ** | (82)  | (82)  |
| **Interest expense ** | - | (7)  |

Part of output Sub-queries:
The value of the Revenues for the three months ended September 30, 2021.
The value of the Revenues for the nine months ended September 30, 2021.
The Cost of revenues for the three months ended September 30, 2021.
......
>>>>>Input:
Query:{}
Table:
{}
>>>>>Output:
""".format(query, table)
    return prompt

def llama_split_prompt(query):
    prompt = """
>>>>>Your Role:
You will be responsible for splitting a query into multiple sub-queries.
>>>>>Your Task:
Given a query, you are now expected to Split that query into finer-grained queries
>>>>>For example:
Query: What are the Loss and Revenues for the three and nine months ended September 30, 2021?

Part of output Sub-queries:
The Loss for the three months ended September 30, 2021.
The Loss for the nine months ended September 30, 2021.
The Loss for the three months ended September 30, 2021.
The Loss for the nine months ended September 30, 2021.
>>>>>Input:
Query:
{}
>>>>>Output:
""".format(query)
    return prompt

def llama_summary_prompt(sentence):
    prompt = """
>>>>>Your Role:
There are multiple values in the sentence, and I'd like you to pick the ones that make sense (not the year, month, day, or serial number, etc.) and explain them, elaborating on what the values correspond to
>>>>>For example:
Sentence:
Gross margin increased to 67.5% from 59.9% in fiscal 2020

Output:
1. 67.5% is the value of Gross margin in 2020
2. 59.9% is the value of Gross margin in 2019

>>>>>Input Sentence:
{}

>>>>>Output:
""".format(sentence)
    return prompt

def summary_value_prompt(sentence):
    prompt = """
>>>>>Your Role:
There are multiple values in the sentence, and I'd like you to pick the ones that make sense (not the year, month, day, or serial number, etc.) and explain them, elaborating on what the values correspond to
I will give you several sentences, Please output the answer with the Serial number.
>>>>>For example:
Sentences 1:
Our effective tax rate in fiscal year 2017 was 35.2%, compared to 34.0% and 36.2% in fiscal years 2016 and 2015, respectively
Sentences 2:
Gross margin increased to 67.5% from 59.9% in fiscal 2020
Sentences 3:
Interest expense for 2022 was $51.1 million, an increase of $13.3 million compared with 2021
Sentences 4:
The Company experienced a foreign currency pre-tax loss of approximately $0.7 million and a pre-tax gain of approximately $1.7 million, respectively, during the three and nine months ended September 30, 2020

Output 1:
1. 35.2% is the value of the effective tax rate in fiscal year 2017
2. 34.0% is the value of the effective tax rate in fiscal year 2016
3. 36.2% is the value of the effective tax rate in fiscal year 2015
Output 2:
1. 67.5% is the value of Gross margin in 2020
2. 59.9% is the value of Gross margin in 2019
Output 3:
1. $51.1 million is the value of Interest expense for 2022
2. $13.3 million is the value of the increase in Interest expense for 2022 compared with 2021
3. $37.8 million is the value of Interest expense for 2021
Output 4:
1. $0.7 million is the value of a foreign currency pre-tax loss during the three and nine months ended September 30, 2020
2. $1.7 million is the value of a pre-tax gain during the three and nine months ended September 30, 2020

>>>>>Input:
{}

>>>>>Output:
""".format(sentence)
    return prompt

def doc_table_prompt(segment, table):
    prompt = """
>>>>>Your Task:
Given a text segment of a financial report and an incomplete table.
You need to supplement the table according to the text, do not change the table structure, just fill the cells.
Returns the table as markdown, with <NEWLINE> as the newline character for the table.
>>>>>Example: 
Financial report 's segment: 
For the company A in 2022Q3 , revenue is $1 .2345 billion; the net income is $50.1245 million. The board members were very satisfied, because they thought company has made great strides this quarter. 
----
Table: 
| Company A | - |<NEWLINE> | 2022 | 2023 | <NEWLINE> | --- | --- | <NEWLINE> | Net income | - | - |<NEWLINE> | Revenue | - | - | <NEWLINE>
----
Updated Table:
| Company A | - |<NEWLINE> | 2022 | 2023 | <NEWLINE> | --- | --- | <NEWLINE> | Net income | $50.1245 million | - | <NEWLINE> | Revenue | $1.2345 billion | - | <NEWLINE>
----
>>>>>Input: 
Financial report 's segment: 
{} 
----
Table: 
{} 
----
>>>>>Output: 
Updated Table:
""".format(segment, table)
    return prompt

def label_box_query_prompt(table):
    prompt = """
>>>>>Your Role:
Your role as an AI data annotation expert is to create queries that dictate the structure and content of data tables. 
>>>>>Your Task:
These queries are key in extracting relevant sentences from documents, with the structure of the tables being defined by the query itself. Your main objective is to formulate precise queries that serve both as tools for data extraction and as blueprints for the table's layout, focusing on the essential elements needed for the table. The challenge is to ensure these queries are comprehensive yet specific, guiding the creation of well-structured and informative tables.The query should be in the form of a question or a requirement
>>>>>Input:
Table:
{}
>>>>>Output (Query):
""".format(table)
    return prompt


def label_box_query_prompt(table):
    prompt = """
>>>>>Your Role:
Your role as an AI data annotation expert is to create one query that dictate the structure of the tables. 
>>>>>Your Task:
These queries are key in extracting relevant sentences from documents, with the structure of the tables being defined by the query itself. Your main objective is to formulate precise queries that serve both as tools for data extraction and as blueprints for the table's layout, focusing on the essential elements needed for the table. The challenge is to ensure these queries are comprehensive yet specific, guiding the creation of well-structured and informative tables.The query should be in the form of a question or a requirement
>>>>>Input:
Table:
{}
>>>>>Output (Query):
""".format(table)
    return prompt

def generate_html_prompt(table):
    prompt = """
>>>>>Your Task:
Convert a markdown table to an html table and determine if the table has a hierarchical structure.
>>>>>Example:
Markdown Table:
|   | **Fiscal Years Ended November 30 ** |   |   |
|   | **2017 ** | **2016 ** | **2015 ** |
| --- | --- | --- | --- |
| **Effective income tax rate ** | 35.2 %  | 34.0 %  | 36.2 %  |

Output:
HTML Table:
<table>
  <thead>  
    <tr>
        <th></th>
        <th colspan="3">Fiscal Years Ended November 30</th>
    </tr>
    <tr>
        <th></th>
        <th>2017</th>
        <th>2016</th>
        <th>2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <th>Effective income tax rate</th>
        <td>35.2%</td>
        <td>34.0%</td>
        <td>36.2%</td>
    </tr>
  </tbody>
</table>

Hierarchical structure: True

>>>>>Input:
{}
>>>>>Output:
""".format(table)
    return prompt

def generate_query_prompt_recall(sentence):
    prompt = """
>>>>>Your Task:
Ask questions about the useful values that appear in this sentence(not the year, month, day, or serial number, etc.), outputting multiple queries, each of which can be about a single value or about multiple values
>>>>>Example:
Sentence:
Comany B's income in 2023 has achieved A very large increase, from $31.2 million to $72.4 million, while company A's income has not increased, but dropped to $43.2 million
Querys:
Please tell me the income of Company B in 2023.
How much did Company B's revenue go up from 2022 to 2023.
Can you show me the income of Company A and Company B in 2023.

>>>>>Input Sentence:
{}
>>>>>Output:
""".format(sentence)
    return prompt

# 生成table的prompt

def simple_prompt(sentences):
    prompt = """
>>>>>Your Task:
Given a query and sentences related to the query.
You need to generate a structured table based on the query and fill in the table according to the relevant sentences.
Returns the table as markdown, with <NEWLINE> as the newline character for the table. Add a split line between header and body, with *** representing the following columns as line names --- representing the table content
>>>>>Example: 
Query:
I want to know the amount in million of loss and revenue of Company A and Company B in 2022 and 2023.
Related sentences: 
1. For the company A in 2022Q3 , revenue is $1.2345 billion; the loss is $50.1245 million. The board members were very satisfied, because they thought company has made great strides this quarter. 
2. Comany B's loss in 2023 has achieved a very large increase, from $31.2 million to $72.4 million, while company A's loss has not increased, but dropped to $43.2 million
3. For the company B in 2022Q3 , revenue is $2.569 billion; the loss is $31.2 million.

Output Table:
| (in million) | Company A | - | Company B | - |<NEWLINE> | | 2022 | 2023 | 2022 | 2023 | <NEWLINE> | *** | --- | --- | --- | --- | <NEWLINE> | Loss | 50.1245 | 43.2 | 31.2 | 72.4 | <NEWLINE> | Revenue | 1,234.5 | - | 2,569 | - | <NEWLINE>

>>>>>Input:
{}

>>>>>Output Table: 
""".format(sentences)
    return prompt

def chain_of_thought_prompt(sentences):
    prompt = """
>>>>>Your Task:
Give you all relevant sentence and the query with the numbering of related sentences.
You need to generate a structured table based on the query and fill in the tabll according to the relevant sentences.
Returns the table as markdown, with <NEWLINE> as the newline character for the table. Add a split line between header and body, with *** representing the following columns as line names --- representing the table content

Process:
1. Cell by Cell, generate table content cell by cell, find the table header and row name for each cell.
For each cell:
    - QUERY BY QUERY, think about which query is most relevant to the row and column names of the cell.
    - Sentence by Sentence, read the sentence associated with this query carefully and think about what information is provided in the sentence that can be filled in the cell, some sentences are useless.
    - Number by Number, focusing on whether or not the number in each sentence is the answer.
2. Output the content of the table, including the content of each cell.

>>>>>Example:
All Sentences:
1.Total revenue during the three months ended March 31, 2023 was $628,425 (comprised of machine sales of $343,000 and non-machine sales of $285,425), compared to the three months ended March 31, 2022 that generated sales of $1,770,075 (comprised of machine sales of $1,033,123 and non-machine sales of $743,952)
2.Total cost of revenue decreased to $405,312 during the three months ended March 31, 2023, compared to the three months ended March 31, 2022 that had a cost of revenue of $1,358,438

Main_Query: What were the values for sales, cost of sales on Three months ended March 31, 2022 and 2023?
Related sentences: [1, 2]
Sub_query: I want to know the Sales on Three months ended March 31, 2022 and 2023
Related sentences: [1]
Sub_query: I want to know the Cost of sales on Three months ended March 31, 2022 and 2023
Related sentences: [2]

Output Table:
|  | Three Months ended March 31 |  |<NEWLINE> |  | 2023 | 2022 |<NEWLINE> | *** | --- | --- |<NEWLINE> | Sales | 628,425  | 1,770,075 |<NEWLINE> | Cost of sales  | 405,312  | 1,358,438  |<NEWLINE> 
>>>>>Input:
{}
>>>>>Output Table:
""".format(sentences)
    return prompt

def unordered_chain_of_thought_prompt(sentences):
    prompt = """
>>>>>Your Task:
Give you a query and some sub-query with some relevant sentences.
You need to generate a structured table based on the query and fill in the tabll according to the relevant sentences.
Returns the table as markdown, with <NEWLINE> as the newline character for the table.

Let's think step by step to generate the table:
1. Query by query, think about the relationship between the query.
2. Sentence by sentence, think about what information is provided by the sentence associated with each query.
3. Cell by cell, when generating the content of the table, do cell by cell generation.

>>>>>Example:
Query and Sub_querys:
I want to know the amount in million of loss and revenue of Company A and Company B in 2022 and 2023.
I want to know the amount in million of loss of Company A in 2022 and 2023
I want to know the amount in million of revenue of Company A in 2022 and 2023
I want to know the amount in million of loss and revenue of Company B in 2022 and 2023
Related sentences:
1. For the company A in 2022Q3 , revenue is $1.2345 billion; the loss is $50.1245 million. The board members were very satisfied, because they thought company has made great strides this quarter. 
2. Comany B's loss in 2023 has achieved a very large increase, from $31.2 million to $72.4 million.
3. In 2023,Company A's loss has not increased, but dropped to $43.2 million
4. For the company B in 2022Q3 , revenue is $2.569 billion; the loss is $31.2 million.

Output Table:
| (in million) | Company A | - | Company B | - |<NEWLINE> | | 2022 | 2023 | 2022 | 2023 | <NEWLINE> | --- | --- | --- | --- | --- | <NEWLINE> | Loss | 50.1245 | 43.2 | 31.2 | 72.4 | <NEWLINE> | Revenue | 1,234.5 | - | 2,569 | - | <NEWLINE>
>>>>>Input:
{}
>>>>>Output Table:
""".format(sentences)
    return prompt

def chain_of_table_prompt(sentences):
    prompt = """
    >>>>>Your Task:
You need to generate a markdown table based on the query and sub-queries with the relevant sentences.
Input: Main query, sub-queries, and relevant sentences.
Returns the table as markdown, with <NEWLINE> as the newline character for the table. Add a split line between header and body, with *** representing the following columns as line names --- representing the table content

[Process Three stages]
First stage is to generate the table structure, including the table header and row names, think step by step:
1. QUERY BY QUERY, think what information the table should contain according to the query, sub-query help you to split the whole query
2. Design carefully table headers and row names, usually the header is related to the time and the table structure may have hierarchical structure in table header and row name. Output the table header and the row names.
3. Specify table dimensions: number of rows and columns. Output the number of rows and columns and split line between header and body.

Second stage is to generate the content of the table, including the content of each cell:
1. According to the query and each sentences related to the query, think about what information should be filled in the table, focusing the numbers in the sentences.
2. Check the table, Ensure each row have same number of cells (count the '|'), ensure the content of each cell is correct.
3. Finalize and output the table.

>>>>>Example:
All Sentences:
1.Total revenue during the three months ended March 31, 2023 was $628,425 (comprised of machine sales of $343,000 and non-machine sales of $285,425), compared to the three months ended March 31, 2022 that generated sales of $1,770,075 (comprised of machine sales of $1,033,123 and non-machine sales of $743,952)
2.Total cost of revenue decreased to $405,312 during the three months ended March 31, 2023, compared to the three months ended March 31, 2022 that had a cost of revenue of $1,358,438
3.Operating expenses during the three months ended March 31, 2023 decreased to $243,623 (comprised of Salaries of $112,887 and Other Sales, Marketing and General and Administrative (\u201cSG&A\u201d) expenses of $130,736), compared to the three months ended March 31, 2022 that produced $417,257 (comprised of Salaries of $267,644 and SG&A expenses of $200 billion)

Main_Query: What were the values for sales, cost of sales, and the Operating expenses including : salaries and wages, other selling general and administrative expenses on Three months ended March 31, 2022 and 2023?
Related sentences: [1, 2, 3]
Sub_query: I want to know the Sales on Three months ended March 31, 2022 and 2023
Related sentences: [1]
Sub_query: I want to know the Cost of sales on Three months ended March 31, 2022 and 2023
Related sentences: [2]

Output:
Header:
|  | Three Months ended March 31 |  |
|  | 2023 | 2022 |
Row name:
| Sales |
| Cost of sales  |
Table Size and Split line:
Col num: 3, Row num: 3, Split line: | *** | --- | --- |
Output Table:
|  | Three Months ended March 31 |  |<NEWLINE> |  | 2023 | 2022 |<NEWLINE> | *** | --- | --- |<NEWLINE> | Sales | 628,425  | 1,770,075 |<NEWLINE> | Cost of sales  | 405,312  | 1,358,438  |<NEWLINE> 
>>>>>Input:
{}
>>>>>Output:
""".format(sentences)
    return prompt

def chain_of_thought_table_prompt(sentences):
    prompt = """
>>>>>Your Task:
You need to generate a markdown table based on the query and sub-queries with the relevant sentences.
Input: Main query, sub-queries, and relevant sentences.
Returns the table as markdown, with <NEWLINE> as the newline character for the table. Add a split line between header and body, with *** representing the following columns as line names --- representing the table content

[Process Three stages]
First stage is to generate the table structure, including the table header and row names, think step by step:
1. QUERY BY QUERY, think what information the table should contain according to the query, sub-query help you to split the whole query
2. Consider carefully the table header and row names, usually the header is related to the time and the table header and row name may have hierarchical structure. Output the table header and row names.
3. For table headers and row names to be revisited, focusing on the hierarchical structure and whether the same content can be extracted as a parent node. Output the hierarchical structure of the table header and then the row names.
4. Determine the size of the table, how many rows and columns. Output the number of rows and columns of the table.

Second stage is to generate the content of the table, including the content of each cell:
1. Cell by Cell, generate table content cell by cell, find the table header and row name for each cell. For each cell, do the 2, 3, 4 step thinking
2. QUERY BY QUERY, think about which query is most relevant to the row and column names of the cell.
3. Sentence by Sentence, read the sentence associated with this query carefully and think about what information is provided in the sentence that can be filled in the cell, some sentences are useless.
4. Number by Number, focusing on whether or not the number in each sentence is the answer.
5. Output the content of the table, including the content of each cell.

Third stage is to refine the table, check that the structure and content of the table are correct.:
1. Check that the table structure has the same number of cells in each row and column.
2. Check that the content of the table is correct, including the correct unit of measure for each cell.
3. Output the refined table.

>>>>>Example:
All Sentences:
1.Total revenue during the three months ended March 31, 2023 was $628,425 (comprised of machine sales of $343,000 and non-machine sales of $285,425), compared to the three months ended March 31, 2022 that generated sales of $1,770,075 (comprised of machine sales of $1,033,123 and non-machine sales of $743,952)
2.Total cost of revenue decreased to $405,312 during the three months ended March 31, 2023, compared to the three months ended March 31, 2022 that had a cost of revenue of $1,358,438
3.Operating expenses during the three months ended March 31, 2023 decreased to $243,623 (comprised of Salaries of $112,887 and Other Sales, Marketing and General and Administrative (\u201cSG&A\u201d) expenses of $130,736), compared to the three months ended March 31, 2022 that produced $417,257 (comprised of Salaries of $267,644 and SG&A expenses of $200 billion)

Main_Query: What were the values for sales, cost of sales, and the Operating expenses including : salaries and wages, other selling general and administrative expenses on Three months ended March 31, 2022 and 2023?
Related sentences: [1, 2, 3]
Sub_query: I want to know the Sales on Three months ended March 31, 2022 and 2023
Related sentences: [1]
Sub_query: I want to know the Cost of sales on Three months ended March 31, 2022 and 2023
Related sentences: [2]
Sub_query: I want to know the Operating expenses including : salaries and wages on Three months ended March 31, 2022 and 2023
Related sentences: [3]
Sub_query: I want to know the Operating expenses including : other selling general and administrative expenses on Three months ended March 31, 2022 and 2023
Related sentences: [3]

Output:
Header:
| Three Months ended March 31, 2023 | Three Months ended March 31, 2022 |
Row name:
| Sales |
| Cost of sales  |
| Operating expenses of Salaries and wages (including contractors) |
| Operating expenses of Other selling general and administrative expenses  |
Hierarchical Structure Header:
| Three Months ended March 31 |  |
|  2023 | 2022 |
Hierarchical Structure Row name:
|  | Sales |
|  | Cost of sales  |
| Operating expenses |  |
|  | Salaries and wages (including contractors) |
|  | Other selling general and administrative expenses  |
Table Size:
Col num: 4, Row num: 7
Output Table:
|  |   | Three Months ended March 31 |  |<NEWLINE> |  | 2023 | 2022 |<NEWLINE> | *** | *** | --- | --- |<NEWLINE> |  | Sales | 628,425  | 1,770,075 |<NEWLINE> |  | Cost of sales  | 405,312  | 1,358,438  |<NEWLINE> | Operating expenses |  |  |<NEWLINE> |  | Salaries and wages (including contractors) | 112,887  | 267,644  |<NEWLINE> |  | Other selling general and administrative expenses  | 130,736  | 200 billion |<NEWLINE> 
Refine Table:
|  |   | Three Months ended March 31 |  |<NEWLINE> |  |   | 2023 | 2022 |<NEWLINE> | *** | *** | --- | --- |<NEWLINE> |  | Sales | 628,425  | 1,770,075 |<NEWLINE> |  | Cost of sales  | 405,312  | 1,358,438  |<NEWLINE> | Operating expenses |  |  |  |<NEWLINE> |  | Salaries and wages (including contractors) | 112,887  | 267,644  |<NEWLINE> |  | Other selling general and administrative expenses  | 130,736  | 200,000 |<NEWLINE> 
>>>>>Input:
{}
>>>>>Output:
""".format(sentences)
    return prompt

def chain_of_thought_table_prompt_v2(sentences):
    prompt = """
>>>>>Your Task:
You need to generate a markdown table based on the query and sub-queries with the relevant sentences.
Input: Main query, sub-queries, and relevant sentences.
Returns the table as markdown, with <NEWLINE> as the newline character for the table. Add a split line between header and body, with *** representing the following columns as line names --- representing the table content

[Process TWO stages]
First stage is to generate the table structure, including the table header and row names, think step by step:
1. QUERY BY QUERY, think what information the table should contain according to the query, sub-query help you to split the whole query
2. Design carefully table headers and row names, usually the header is related to the time and the table structure may have hierarchical structure in table header and row name. Output the table header and the row names.
3. Specify table dimensions: number of rows and columns. Output the number of rows and columns and split line between header and body.

Second stage is to generate the content of the table, including the content of each cell:
1. Cell by Cell, generate table content cell by cell, find the table header and row name for each cell.
For each cell:
    - QUERY BY QUERY, think about which query is most relevant to the row and column names of the cell.
    - Sentence by Sentence, read the sentence associated with this query carefully and think about what information is provided in the sentence that can be filled in the cell, some sentences are useless.
    - Number by Number, focusing on whether or not the number in each sentence is the answer.
2. Check the table, Ensure each row have same number of cells (count the '|'), ensure the content of each cell is correct.
3. Finalize and output the table.

>>>>>Example:
All Sentences:
1.Total revenue during the three months ended March 31, 2023 was $628,425 (comprised of machine sales of $343,000 and non-machine sales of $285,425), compared to the three months ended March 31, 2022 that generated sales of $1,770,075 (comprised of machine sales of $1,033,123 and non-machine sales of $743,952)
2.Total cost of revenue decreased to $405,312 during the three months ended March 31, 2023, compared to the three months ended March 31, 2022 that had a cost of revenue of $1,358,438

Main_Query: What were the values for sales, cost of sales on Three months ended March 31, 2022 and 2023?
Related sentences: [1, 2]
Sub_query: I want to know the Sales on Three months ended March 31, 2022 and 2023
Related sentences: [1]
Sub_query: I want to know the Cost of sales on Three months ended March 31, 2022 and 2023
Related sentences: [2]

Output:
Header:
|  | Three Months ended March 31 |  |
|  | 2023 | 2022 |
Row name:
| Sales |
| Cost of sales  |
Table Size and Split line:
Col num: 3, Row num: 3, Split line: | *** | --- | --- |
Output Table:
|  | Three Months ended March 31 |  |<NEWLINE> |  | 2023 | 2022 |<NEWLINE> | *** | --- | --- |<NEWLINE> | Sales | 628,425  | 1,770,075 |<NEWLINE> | Cost of sales  | 405,312  | 1,358,438  |<NEWLINE> 
>>>>>Input:
{}
>>>>>Output:
""".format(sentences)
    return prompt

def html_simple_prompt(sentences):
    prompt = """
>>>>>Your Task:
Given a query and sentences related to the query.
You need to generate a structured table based on the query and fill in the table according to the relevant sentences.
Returns the table tables in the html format: Using <table> <thead> <tbody> <tr> <th> <td> and rowspan , colspan for the hierarchical structure of the table
>>>>>Example: 
Query:
I want to know the amount in million of loss and revenue of Company A and Company B in 2022 and 2023.
Related sentences: 
1. For the company A in 2022Q3 , revenue is $1.2345 billion; the loss is $50.1245 million. The board members were very satisfied, because they thought company has made great strides this quarter. 
2. Comany B's loss in 2023 has achieved a very large increase, from $31.2 million to $72.4 million, while company A's loss has not increased, but dropped to $43.2 million
3. For the company B in 2022Q3 , revenue is $2.569 billion; the loss is $31.2 million.

Output Table:
<table>
<thead>
<tr><th>(in million)</th><th colspan=2>Company A</th><th colspan=2>Company B</th></tr>
<tr><th></th><th>2022</th><th>2023</th><th>2022</th><th>2023</th></tr>
</thead>
<tbody>
<tr><th>Loss</th><td>50.1245</td><td>43.2</td><td>31.2</td><td>72.4</td></tr>
<tr><th>Revenue</th><td>1,234.5</td><td>-</td><td>2,569</td><td>-</td></tr>
</tbody>
</table>
>>>>>Input:
{}
>>>>>Output Table: 
""".format(sentences)
    return prompt


def improving_html_simple_prompt(sentences):
    prompt = """
>>>>>Your Task:
Given a query and sentences related to the query.
You need to generate a structured table based on the query and fill in the table according to the relevant sentences.
Returns the table tables in the following improving-html format:
1. Replace <tr> with <hr> at the beginning of each line of the table header, <hr> means table header row.
2. Replace <tr> with <br> at the beginning of each line of the table body, <br> means table body row.
3. Cell attribute have three type: a. cell type(th or td) b. cell span(colspan or rowspan) c. cell coordinates( Multiple rows or columns using | to show (row_id, col_id1|col_id2)). So the composition of the unit attributes is <a b c> like <th colspan=2 (1, 2|3)>

>>>>>Example: 
Query:
I want to know the amount in million of loss and revenue of Company A and Company B in 2022 and 2023.
Related sentences: 
1. For the company A in 2022Q3 , revenue is $1.2345 billion; the loss is $50.1245 million. The board members were very satisfied, because they thought company has made great strides this quarter. 
2. Comany B's loss in 2023 has achieved a very large increase, from $31.2 million to $72.4 million, while company A's loss has not increased, but dropped to $43.2 million
3. For the company B in 2022Q3 , revenue is $2.569 billion; the loss is $31.2 million.

Output Table:
<table>
<thead>
<hr (1,0)><th (1,1)>(in million)</th><th colspan=2 (1, 2|3)>Company A</th><th colspan=2 (1,4|5)>Company B</th></hr>
<hr (2,0)><th (2,1)></th><th (2,2)>2022</th><th (2,3)>2023</th><th (2,4)>2022</th><th (2,5)>2023</th></hr>
</thead>
<tbody>
<br (3,0)><th (3,1)>Loss</th><td (3,2)>50.1245</td><td (3,3)>43.2</td><td (3,4)>31.2</td><td (3,5)>72.4</td></br>
<br (4,0)><th (4,1)>Revenue</th><td (4,2)>1,234.5</td><td (4,3)>-</td><td (4,4)>2,569</td><td (4,5)>-</td></br>
</tbody>
</table>
>>>>>Input:
{}
>>>>>Output Table: 
""".format(sentences)
    return prompt

def html_chain_of_thought_table_prompt(sentences):
    prompt = """
>>>>>Your Task:
You need to generate a html table based on the query and sub-queries with the relevant sentences.
Input: Main query, sub-queries, and relevant sentences.
Returns the table tables in the html format: Using <table> <thead> <tbody> <tr> <th> <td> and rowspan, colspan for the hierarchical structure of the table

[Process TWO stages]
First stage is to generate the table structure, including the table header and row names, think step by step:
1. QUERY BY QUERY, think what information the table should contain according to the query, sub-query help you to split the whole query
2. Design carefully table headers and row names, usually the header is related to the time and the table structure may have hierarchical structure in table header and row name. Output the table header and the row names.
3. Specify table dimensions: number of rows and columns. Output the number of rows and columns.

Second stage is to generate the content of the table, including the content of each cell:
1. Cell by Cell, generate table content cell by cell, find the table header and row name for each cell.
For each cell:
    - QUERY BY QUERY, think about which query is most relevant to the row and column names of the cell.
    - Sentence by Sentence, read the sentence associated with this query carefully and think about what information is provided in the sentence that can be filled in the cell, some sentences are useless.
    - Number by Number, focusing on whether or not the number in each sentence is the answer.
2. Check the table, Ensure each row have same number of cells, ensure the content of each cell is correct.
3. Finalize and output the table.

>>>>>Example:
All Sentences:
1.Total revenue during the three months ended March 31, 2023 was $628,425 (comprised of machine sales of $343,000 and non-machine sales of $285,425), compared to the three months ended March 31, 2022 that generated sales of $1,770,075 (comprised of machine sales of $1,033,123 and non-machine sales of $743,952)
2.Total cost of revenue decreased to $405,312 during the three months ended March 31, 2023, compared to the three months ended March 31, 2022 that had a cost of revenue of $1,358,438
3.Operating expenses during the three months ended March 31, 2023 decreased to $243,623 (comprised of Salaries of $112,887 and Other Sales, Marketing and General and Administrative (\u201cSG&A\u201d) expenses of $130,736), compared to the three months ended March 31, 2022 that produced $417,257 (comprised of Salaries of $267,644 and SG&A expenses of $200 billion)

Main_Query: What were the values for sales, cost of sales, and the Operating expenses including : salaries and wages, other selling general and administrative expenses on Three months ended March 31, 2022 and 2023?
Related sentences: [1, 2, 3]
Sub_query: I want to know the Sales on Three months ended March 31, 2022 and 2023
Related sentences: [1]
Sub_query: I want to know the Cost of sales on Three months ended March 31, 2022 and 2023
Related sentences: [2]
Sub_query: I want to know the Operating expenses including : salaries and wages on Three months ended March 31, 2022 and 2023
Related sentences: [3]
Sub_query: I want to know the Operating expenses including : other selling general and administrative expenses on Three months ended March 31, 2022 and 2023
Related sentences: [3]

Output:
Header:
<thead>
<tr><th></th><th colspan=2>Three Months ended March 31</th></tr>
<tr><th></th><th>2023</th><th>2022</th></tr>
</thead>
Row name:
<tbody>
<tr><th>Sales</th></tr>
<tr><th>Cost of sales</th></tr>
<tr><th rowspan=3>Operating expenses</th></tr>
<tr><th>Salaries and wages (including contractors)</th></tr>
<tr><th>Other selling general and administrative expenses</th></tr>
</tbody>
Table Size:
Col num: 3, Row num: 7
Output Table:
<table>
<thead>
<tr><th></th><th colspan=2>Three Months ended March 31</th></tr>
<tr><th></th><th>2023</th><th>2022</th></tr>
</thead>
<tbody>
<tr><th>Sales</th><th>628,425</th><th>1,770,075</th></tr>
<tr><th>Cost of sales</th><th>405,312</th><th>1,358,438</th></tr>
<tr><th rowspan=3>Operating expenses</th>/tr>
<tr><th>Salaries and wages (including contractors)</th><th>112,887</th><th>267,644</th></tr>
<tr><th>Other selling general and administrative expenses</th><th>130,736</th><th>200,000</th></tr>
</tbody>
</table>
>>>>>Input:
{}
>>>>>Output:
""".format(sentences)
    return prompt

def generate_query_prompt(table):
    prompt = """
>>>>>Your Task:
Give you a table, you need to generate a query based on the table header and row name. The query should include all information of the table.
>>>>>Example 1:
Table:
| (in million) | Company A |  | Company B |  |
| | 2022 | 2023 | 2022 | 2023 |
| ----------------- |
| Loss |  |  |  |  |
| Revenue |  |  |  |  |
Output Query:
I want to know the amount in million of loss and revenue of Company A and Company B in 2022 and 2023.
>>>>>Example 2:
Table:
|   | **Three Months Ended September 30, ** | **Nine Months Ended September 30, ** |
|   | **2021 ** | **2021 ** |
| ----------------- |
| **Revenues ** |   |   |
| **Cost of revenues ** |   |   |
| **Gross profit ** |   |   |
| **Interest expense ** |  |   |
Output Query:
Can you tell me the revenues, cost of revenues, gross profit and interest expense of three months ended September 30, 2021 and nine months ended September 30, 2021?
>>>>>Input:
{}
>>>>>Output Query:
""".format(table)
    return prompt

# evaluate prompt
def markdown_evaluate_prompt(table_type, pred_table, gold_table):
    prompt = """
>>>>>Your Task:
We want to evaluate how similar the following {} tables are. Please give two similarity score, one for the table structure and one for the table content. 
Score the "header content similarity", "body content similarity" and "structural similarity" between 0 and 10. 
- Header Content similarity: Focus on the column name and row name, 10 if the tables have same header and row name with same hierarchy structure. If about 50% of the cells have the same data, the score should be 5.
- Body Content similarity: Focus on the table body cell, 10 if the contents of the table body cells are identical, 0 if they are entirely different. If about 50% of the cells have the same data, the score should be 5.
- Structural similarity: You can ignore the difference in order by focusing only on the headers and row names, by thinking of the table structure as a tree with two implied parents, a row node and a column node, and then by building the tree. 10 if the tables have same header and row name with same hierarchy structure.
Output a dict object such as the following:
{{
    "header_content_similarity": 5,
    "body_content_similarity": 5,
    "structural_similarity": 10
}}
Think carefully, and then output the scores.
>>>>>Example:
Table 1:
| (in million) | Company A |  | Company B |  |
| | 2022 | 2023 | 2022 | 2023 |
| *** | --- | --- | --- | --- |
| Loss | 1 | 2 | 3 | 4 |
| Revenue | 1 | 2 | 3 | 4 |
Table 2:
| (in million) | Company A 2022 | Company A 2023 | Company B 2022 | Company B 2023 |
| *** | --- | --- | --- | --- |
| Loss | 1 | 4 | 3 | 4 |
| Revenue | 1 | 7 | 3 | 4 |
Output:
{{
    "header_content_similarity": 8,
    "body_content_similarity": 7.5,
    "structural_similarity": 8
}}
----------------
Table 1:
| (in million) | Company A |  | Company B |  |
| | 2022 | 2023 | 2022 | 2023 |
| *** | --- | --- | --- | --- |
| Loss | 1 | 2 | 3 | 4 |
| Revenue | 1 | 2 | 3 | 4 |
Table 2:
| (in million) | Company A 2022 | Company A 2023 | Company B 2022 | Company B 2023 |
| *** | --- | --- | --- | --- |
| Loss | 2 | 4 | 3 | 4 |
Output:
{{
    "header_content_similarity": 5.5,
    "body_content_similarity": 2.5,
    "structural_similarity": 6
}}
>>>>>Input:
Table 1:
{}
Table 2:
{}
>>>>>Output:
""".format(table_type, pred_table, gold_table)
    return prompt

def html_evaluate_prompt(table_type, pred_table, gold_table):
    prompt = """
>>>>>Your Task:
We want to evaluate how similar the following {} tables are. Please give two similarity score, one for the table structure and one for the table content. 
Score the "header content similarity", "body content similarity" and "structural similarity" between 0 and 10. 
- Header Content similarity: Focus on the column name and row name, 10 if the tables have same header and row name with same hierarchy structure. If about 50% of the cells have the same data, the score should be 5.
- Body Content similarity: Focus on the table body cell, 10 if the contents of the table body cells are identical, 0 if they are entirely different. If about 50% of the cells have the same data, the score should be 5.
- Structural similarity: You can ignore the difference in order by focusing only on the headers and row names, by thinking of the table structure as a tree with two implied parents, a row node and a column node, and then by building the tree. 10 if the tables have same header and row name with same hierarchy structure.
Output a dict object such as the following:
{{
    "content_similarity": 5,
    "structural_similarity": 10
}}
Think carefully, and then output the scores.
>>>>>Example:
Table 1:
<table>
    <tr>
        <th>(in million)</th>
        <th colspan=2>Company A</th>
        <th colspan=2>Company B</th>
    </tr>
    <tr>
        <td></td>
        <td>2022</td>
        <td>2023</td>
        <td>2022</td>
        <td>2023</td>
    </tr>
    <tr>
        <th>Loss</th>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
    </tr>
    <tr>
        <th>Revenue</th>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
    </tr>
</table>
Table 2:
<table>
    <tr>
        <th>(in million)</th>
        <th>Company A</th>
        <th></th>
        <th>Company B</th>
        <th></th>
    </tr>
    <tr>
        <td></td>
        <td>2022</td>
        <td>2023</td>
        <td>2022</td>
        <td>2023</td>
    </tr>
    <tr>
        <th>Loss</th>
        <td>1</td>
        <td>3</td>
        <td>3</td>
        <td>4</td>
    </tr>
    <tr>
        <th>Revenue</th>
        <td>3</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
    </tr>
</table>
Output:
{{
    "header_content_similarity": 5,
    "body_content_similarity": 7.5,
    "structural_similarity": 9
}}
----------------
Table 1:
<table>
    <tr>
        <th>(in million)</th>
        <th colspan=2>Company A</th>
        <th colspan=2>Company B</th>
    </tr>
    <tr>
        <td></td>
        <td>2022</td>
        <td>2023</td>
        <td>2022</td>
        <td>2023</td>
    </tr>
    <tr>
        <th>Loss</th>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
    </tr>
    <tr>
        <th>Revenue</th>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
    </tr>
</table>
Table 2:
<table>
    <tr>
        <th>(in million)</th>
        <th colspan=2>Company A</th>
    </tr>
    <tr>
        <td></td>
        <td>2022</td>
        <td>2023</td>
    </tr>
    <tr>
        <th>Loss</th>
        <td>1</td>
        <td>2</td>
    </tr>
</table>
Output:
{{
    "header_content_similarity": 2.5,
    "body_content_similarity": 2.5,
    "structural_similarity": 5
}}
>>>>>Input:
Table 1:
{}
Table 2:
{}
>>>>>Output:
""".format(table_type, pred_table, gold_table)
    return prompt

def ttt_prompt(sentences):
    prompt = """
>>>>>Your Task:
Extract the Player table from sentence, Form a header based on the content of the sentence, the header is not necessarily the same as the example.
Returns the table as markdown, with <NEWLINE> as the newline character for the table.
>>>>>Example: 
Example 1:
Sentence:
Isaiah Thomas dished out a career - best 15 assists, while also scoring a game - high 29 points. His standout effort led to a stellar shooting night for Boston, which posted respective success rates of 55 and 55 percent from the field and three - point range. Al Horford and Jae Crowder had matching 21 - point efforts. Avery Bradley compiled 14 points, Jaylen Brown led the second unit with 10, and Marcus Smart went for nine points over 28 minutes despite coming in questionable with an illness. Gordon Hayward's 23 points led Utah, and Derrick Favors had 12 points over 29 minutes. Rodney Hood supplied nine points, while Rudy Gobert brought down 13 rebounds. Joe Johnson and Boris Diaw went for 17 and 15 points off the bench, respectively, with the latter impressively compiling his total in just 15 minutes.

Output Table:
|  | Assists | Minutes played | Points | Total rebounds | <NEWLINE> | Gordon Hayward |  |  | 23 |  | <NEWLINE> | Derrick Favors |  | 29 | 12 |  | <NEWLINE> | Rudy Gobert |  |  |  | 13 | <NEWLINE> | Rodney Hood |  |  | 9 |  | <NEWLINE> | Joe Johnson |  |  | 17 |  | <NEWLINE> | Boris Diaw |  | 15 | 15 |  | <NEWLINE> | Jae Crowder |  |  | 21 |  | <NEWLINE> | Al Horford |  |  | 21 |  | <NEWLINE> | Avery Bradley |  |  | 14 |  | <NEWLINE> | Isaiah Thomas | 15 |  | 29 |  | <NEWLINE> | Marcus Smart |  | 28 | 9 |  | <NEWLINE> | Jaylen Brown |  |  | 10 |  | <NEWLINE>

Example 2:
Sentence:
On Sunday, rising young forward Doug McDermott had one of the better nights of his career. McDermott scored a team - high 31 points with three three - pointers and 10 - of - 11 from the free - throw line. Superstar forward Jimmy Butler anchored the starting five, going for 16 points, eight rebounds, six assists, and three steals. Veteran big man Taj Gibson led the starting five in scoring with 18 points and had eight rebounds. As a team, the Bulls shot a whopping 51 percent from the field and reached the free - throw line 35 times. The Grizzlies, meanwhile, saw strong play from their veteran core. Center Marc Gasol scored 24 points to go with 11 rebounds. Point guard Mike Conley, meanwhile, led the team with 28 points and eight assists. Off the bench, big man Zach Randolph had 16 rebounds, including eight on the offensive end.

Output Table:
|  | Assists | Blocks | Defensive rebounds | 3-pointers attempted | 3-pointers made | Free throws attempted | Free throws made | Offensive rebounds | Points | Total rebounds | Steals | Turnovers | <NEWLINE> | Taj Gibson |  |  |  |  |  | 5 |  |  | 18 | 8 |  |  | <NEWLINE> | Jimmy Butler | 6 |  |  | 5 |  |  |  |  | 16 | 8 | 3 |  | <NEWLINE> | Doug McDermott |  | 1 |  |  | 3 | 11 | 10 |  | 31 |  | 1 | 1 | <NEWLINE> | Marc Gasol |  |  |  |  |  |  |  |  | 24 | 11 |  |  | <NEWLINE> | Mike Conley | 8 |  |  |  |  |  |  |  | 28 |  |  |  | <NEWLINE> | Zach Randolph |  |  | 8 |  |  |  |  | 8 |  | 16 |  |  | <NEWLINE> 
>>>>>Input:
{}

>>>>>Output Table: 
""".format(sentences)
    return prompt

def ttt_tabtalk_prompt(sentences):
    prompt = """
>>>>>Your Task:
Extract the Player table from sentence. Form a header based on the content of the sentence, the header is not necessarily the same as the example.
Returns the table as markdown, with <NEWLINE> as the newline character for the table.

[Process TWO stages]
First stage is to generate the table structure, including the table header and row names, think step by step:
1. Read the sentence, think what information the table should contain.
2. Design carefully table headers and row names, often row name is players name. Output the table header and the row names.
3. Specify table dimensions: number of rows and columns. Output the number of rows and columns.

Second stage is to generate the content of the table, including the content of each cell:
1. Cell by Cell, generate table content cell by cell, find the table header and row name for each cell.
For each cell:
    - Sentence by Sentence, think about what information is provided in the sentence that can be filled in the cell.
    - Number by Number, focusing on whether or not the number in each sentence is the answer.
2. Check the table, Ensure each row have same number of cells (count the '|'), ensure the content of each cell is correct.
3. Finalize and output the table.

>>>>>Example:
Example 1:
Sentence:
Isaiah Thomas dished out a career - best 15 assists, while also scoring a game - high 29 points. His standout effort led to a stellar shooting night for Boston, which posted respective success rates of 55 and 55 percent from the field and three - point range. Al Horford and Jae Crowder had matching 21 - point efforts. Avery Bradley compiled 14 points, Jaylen Brown led the second unit with 10, and Marcus Smart went for nine points over 28 minutes despite coming in questionable with an illness. Gordon Hayward's 23 points led Utah, and Derrick Favors had 12 points over 29 minutes. Rodney Hood supplied nine points, while Rudy Gobert brought down 13 rebounds. Joe Johnson and Boris Diaw went for 17 and 15 points off the bench, respectively, with the latter impressively compiling his total in just 15 minutes.

Output:
Header:
|  | Assists | Minutes played | Points | Total rebounds | <NEWLINE> 
Row name:
| Gordon Hayward |
| Derrick Favors |
| Rudy Gobert |
| Rodney Hood |
| Joe Johnson |
| Boris Diaw |
| Jae Crowder |
| Al Horford |
| Avery Bradley |
| Isaiah Thomas |
| Marcus Smart |
| Jaylen Brown |
Table Size:
Col num: 5, Row num: 13
Output Table:
|  | Assists | Minutes played | Points | Total rebounds | <NEWLINE> | Gordon Hayward |  |  | 23 |  | <NEWLINE> | Derrick Favors |  | 29 | 12 |  | <NEWLINE> | Rudy Gobert |  |  |  | 13 | <NEWLINE> | Rodney Hood |  |  | 9 |  | <NEWLINE> | Joe Johnson |  |  | 17 |  | <NEWLINE> | Boris Diaw |  | 15 | 15 |  | <NEWLINE> | Jae Crowder |  |  | 21 |  | <NEWLINE> | Al Horford |  |  | 21 |  | <NEWLINE> | Avery Bradley |  |  | 14 |  | <NEWLINE> | Isaiah Thomas | 15 |  | 29 |  | <NEWLINE> | Marcus Smart |  | 28 | 9 |  | <NEWLINE> | Jaylen Brown |  |  | 10 |  | <NEWLINE>

Example 2:
Sentence:
On Sunday, rising young forward Doug McDermott had one of the better nights of his career. McDermott scored a team - high 31 points with three three - pointers and 10 - of - 11 from the free - throw line. Superstar forward Jimmy Butler anchored the starting five, going for 16 points, eight rebounds, six assists, and three steals. Veteran big man Taj Gibson led the starting five in scoring with 18 points and had eight rebounds. As a team, the Bulls shot a whopping 51 percent from the field and reached the free - throw line 35 times. The Grizzlies, meanwhile, saw strong play from their veteran core. Center Marc Gasol scored 24 points to go with 11 rebounds. Point guard Mike Conley, meanwhile, led the team with 28 points and eight assists. Off the bench, big man Zach Randolph had 16 rebounds, including eight on the offensive end.

Outpt:
Header:
|  | Assists | Blocks | Defensive rebounds | 3-pointers attempted | 3-pointers made | Free throws attempted | Free throws made | Offensive rebounds | Points | Total rebounds | Steals | Turnovers | <NEWLINE> 
Row name:
| Taj Gibson |
| Jimmy Butler |
| Doug McDermott |
| Marc Gasol |
| Mike Conley |
| Zach Randolph |
Table Size:
Col num: 13, Row num: 6
Output Table:
|  | Assists | Blocks | Defensive rebounds | 3-pointers attempted | 3-pointers made | Free throws attempted | Free throws made | Offensive rebounds | Points | Total rebounds | Steals | Turnovers | <NEWLINE> | Taj Gibson |  |  |  |  |  | 5 |  |  | 18 | 8 |  |  | <NEWLINE> | Jimmy Butler | 6 |  |  | 5 |  |  |  |  | 16 | 8 | 3 |  | <NEWLINE> | Doug McDermott |  | 1 |  |  | 3 | 11 | 10 |  | 31 |  | 1 | 1 | <NEWLINE> | Marc Gasol |  |  |  |  |  |  |  |  | 24 | 11 |  |  | <NEWLINE> | Mike Conley | 8 |  |  |  |  |  |  |  | 28 |  |  |  | <NEWLINE> | Zach Randolph |  |  | 8 |  |  |  |  | 8 |  | 16 |  |  | <NEWLINE> 
>>>>>Input:
{}
>>>>>Output:
""".format(sentences)
    return prompt

def query_table_prompt(sentences):
    prompt = """
>>>>>Your Task:
Given a query and Document sentences.
You need to generate a structured table based on the query and fill in the table according to the Document sentences.
Returns the table in the html format: Using <table> <thead> <tbody> <tr> <th> <td> and rowspan , colspan for the hierarchical structure of the table
>>>>>Example: 
Query:
I want to know the amount in million of loss and revenue of Company A and Company B in 2022 and 2023.
Document sentences: 
1. For the company A in 2022Q3 , revenue is $1.2345 billion; the loss is $50.1245 million. The board members were very satisfied, because they thought company has made great strides this quarter. 
2. Comany B's loss in 2023 has achieved a very large increase, from $31.2 million to $72.4 million, while company A's loss has not increased, but dropped to $43.2 million
3. For the company B in 2022Q3 , revenue is $2.569 billion; the loss is $31.2 million.

Output Table:
<table>
<thead>
<tr><th>(in million)</th><th colspan=2>Company A</th><th colspan=2>Company B</th></tr>
<tr><th></th><th>2022</th><th>2023</th><th>2022</th><th>2023</th></tr>
</thead>
<tbody>
<tr><th>Loss</th><td>50.1245</td><td>43.2</td><td>31.2</td><td>72.4</td></tr>
<tr><th>Revenue</th><td>1,234.5</td><td>-</td><td>2,569</td><td>-</td></tr>
</tbody>
</table>
>>>>>Input:
{}
>>>>>Output Table: 
""".format(sentences)
    return prompt

def query_table_updated_prompt(sentences):
    prompt = """
>>>>>Your Task:
Update the table already generated based on query and partial sentences
Give you the query and the rest of the document sentences
You need to update the structured table based on the query and fill in the table according to the document sentences.
Returns the table tables in the html format: Using <table> <thead> <tbody> <tr> <th> <td> and rowspan , colspan for the hierarchical structure of the table
>>>>>Example: 
Table:
| | Company A | - |<NEWLINE> | | 2022 | 2023 | <NEWLINE> | --- | --- | --- | <NEWLINE> | Net income | $50.1245 million | $43.2 million | <NEWLINE> | Revenue | $1.2345 billion | - | <NEWLINE>

Query:
I want to know the income and revenue of Company A and Company B in 2022 and 2023

Related sentences: 
1. For the company B in 2022Q3 , revenue is $2.569 billion; the net income is $31.2 million.
2. Comany B's income in 2023 has achieved A very large increase, from $31.2 million to $72.4 million, while company A's income has not increased, but dropped to $43.2 million

Updated Table:
| | Company A | - | Company B | - |<NEWLINE> | | 2022 | 2023 | 2022 | 2023 | <NEWLINE> | --- | --- | --- | --- | --- | <NEWLINE> | Net income | $50.1245 million | $43.2 million | $31.2 million | $72.4 million| <NEWLINE> | Revenue | $1.2345 billion | - | $2.569 billion | - | <NEWLINE>

>>>>>Input:
Table:
{}
Query: 
{}
Related sentences:
{}

>>>>>Output Updated Table: 
""".format(table, query, sentences)
    return prompt