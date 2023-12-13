# README

EN: [English README](README.md)

## 概述

本项目对 qwen-14b-chat、baichuan2-13b-chat-v1、gpt-3.5-turbo 三个预训练大模型进行测试，测试任务包括：代码生成、多轮对话角色扮演、数学推理、文案撰写。

项目提出了各任务下的表现最好的 final system prompt (同一任务下，不同模型使用相同 system prompt)，并予以完整的设计过程。每个任务中 final system prompt 的有效性在实验部分通过详细的表格、折线图等结果对比给出。另外，本项目也定性地讨论了各模型的能力边界、缺陷、风险，并提出其他发现、大模型评估大模型可能性。

本项目旨在帮助想要进行类似任务的开发者及测试人员，提供较好的测试平台出发点与实验流程。

详细的项目任务定义、测试结果、技术报告，见 [report](report.pdf)。(注意：报告为中文)


## 使用

conda 环境依赖见 [./requirements.txt](./requirements.txt)。

### 文件结构

注意：本项目需要使用你的 [OpenAI API-Key](https://platform.openai.com/docs/overview)、[DashScope API-Key](https://dashscope.aliyun.com/)。请新建 `./.env` 文件并按如下填写：

```python
# Once you add your API key below, make sure to not share it with anyone! The API key should remain private.
OPENAI_API_KEY=sk-XXXXXXX
DASHSCOPE_API_KEY=sk-YYYYYYY
```

需保持如下文件结构：

```sh
.
├── data
│   ├── judge-prompts.jsonl
│   └── question-prompts.jsonl
├── response
│   ├── code_snippets
│   │   └── # 代码生成任务 pass@3 测试时产生的 Python 文件	(.py)
│   ├── raw_responses
│   │   └── # 本次请求模型的响应	(.json)
│   └── raw_responses_backups
│       ├── default_prompt
│       │   └── # default prompt 下各任务各模型响应记录	(.json)
│       └── final_prompt
│           └── # final prompt 下各任务各模型响应记录	(.json)
├── result
│   ├── gpt_raw_evals
│   │   └── # 本次评估的响应	(.json)
│   ├── gpt_raw_evals_backups
│   │   ├── default_prompt
│   │   │   └── # default prompt 下各任务各模型评估记录	(.json)
│   │   └── final_prompt
│   │       └── # final prompt 下各任务各模型评估记录	(.json)
│   └── reuslt.ipynb
└── src
    ├── coding.py
    ├── main.py
    ├── math_analysis.py
    ├── roleplay.py
    ├── utils.py
    └── writing.py
```

### 数据说明

项目提供了 `seed = 2024` (代码生成任务中三个 pass 下 `seed = 2023, 2024, 2025`)下，各任务、各模型在 default system prompt、final system prompt 下的响应、gpt-3.5-turbo 的评价。

- **模型响应:**
  - `./response/raw_responses_backups/default_prompt`。
  -  `./response/raw_responses_backups/final_prompt`。
- **gpt-3.5-turbo 的评价:**
  -  `./results/gpt_raw_evals_backups/default_prompt`。
  -  `./results/gpt_raw_evals_backups/final_prompt`。

以下部分中: `TASK` 可以是 `"coding"`、`"math"`、`"roleplay"`、`"writing"`; `MODEL` 可以是 `"qwen-14b-chat"`、`"baichuan2-13b-chat-v1"`、`"gpt-3.5-turbo"`。具体项目任务定义、测试结果、技术报告，见 [report](report.pdf)。(注意：报告为中文)

**快速直接查看评分 (包括代码生成任务的 pass@3 指标):**

```sh
python src/main.py --task TASK --model MODEL --eval --debug
```

将查看存放在 `./results/gpt_raw_evals` 下对应任务、模型的评估结果的文件。因此，若想查看之前任何一次输出记录，可从上述文件夹中取出对应文件到 `./results/gpt_raw_evals` 再运行。

### 代码说明

项目提供了完整的代码，可以直接使用。

- **请求模型响应:** 将通过 API 调用模型，question 范围为 `./data/pj3-test.jsonl` 中所有本任务的 question。响应将保存在 `./response/raw_responses`。

  ```sh
  python src/main.py --task TASK --model MODEL --request
  # Example
  python src/main.py --task "math" --model "gpt-3.5-turbo" --request
  ```

- **查看模型响应:** 格式化输出 `./response/raw_responses`。主要用于调试，意义不大。

  ```sh
  python src/main.py --task TASK --model MODEL --request --debug
  ```


- **请求 gpt-3.5-turbo 评价 (若代码生成任务，将包括 pass@3 测试):**

  ```sh
  python src/main.py --task TASK --model MODEL --eval
  ```

- **也可以自动串行运行:** 注意，需要避免不同模型同时测试，会引起 gpt-3.5-turbo 请求超额 (3 Request / min)。代码中只对同一模型的请求做了停等、二进制指数退避等处理，但未对不同模型同时测试做处理。

  ```sh
  python src/main.py --task TASK --model MODEL --request --eval
  ```

- **删除临时代码文件:** 代码生成任务请求 gpt-3.5-turbo 评价时，会在 `./response/code_snippets/MODEL` 下生成代码文件，创建子进程并运行。可以一键清空：

  ```sh
  python src/main.py --task TASK --model MODEL --clean_files --clean_response_messages
  ```
