# README

CH: [中文 README](README_zh.md)

## Description

This project tested three pre-trained large language models, qwen-14b-chat, baichuan2-13b-chat-v1, and gpt-3.5-turbo, with tasks including code generation, multi-turn dialogue role-playing, mathematical reasoning, and copywriting.

The project proposed the best-performing final system prompts for each task (different models using the same system prompt under the same task) and provided a comprehensive design process.

The effectiveness of the final system prompt in each task is demonstrated in the experimental section through detailed comparisons of results such as tables and line graphs. In addition, this project also qualitatively discusses the capabilities, limitations, and risks of each model, and presents other findings and the potential for large-scale model evaluation.

The aim of this project is to provide developers and testers who want to perform similar tasks with a good starting point and experimental process for testing.

For detailed project task description, test results and technical reports, please see the [report](report.pdf). (NOTICE: The report is in Chinese. *If you would like to help with the translation, please submit a pull request. Thank you!* 🥰).


## Usage

conda environment dependencies can be found in [./requirements.txt](./requirements.txt).


### File Structure

NOTICE: This project requires the use of your [OpenAI API-Key](https://platform.openai.com/docs/overview) and [DashScope API-Key](https://dashscope.aliyun.com/). Please create a new `./.env` file and fill in accordance with the following:

```python
# Once you add your API key below, make sure to not share it with anyone! The API key should remain private.
OPENAI_API_KEY=sk-XXXXXXX
DASHSCOPE_API_KEY=sk-YYYYYYY
```

The following file structure must be maintained:

```sh
.
├── data
│   ├── judge-prompts.jsonl
│   └── queston-prompts.jsonl
├── response
│   ├── code_snippets
│   │   └── # the temporary Python files generated during pass@3 testing of code generation tasks	(.py)
│   ├── raw_responses
│   │   └── # the response of the model requested this time	(.json)
│   └── raw_responses_backups
│       ├── default_prompt
│       │   └── # the response records of each task and model under default prompt	(.json)
│       └── final_prompt
│           └── # the response records of each task and model under final prompt	(.json)
├── result
│   ├── gpt_raw_evals
│   │   └── # the response of the evaluation this time	(.json)
│   ├── gpt_raw_evals_backups
│   │   ├── default_prompt
│   │   │   └── # the evaluation records of each task and model under default prompt	(.json)
│   │   └── final_prompt
│   │       └── # the evaluation records of each task and model under final prompt	(.json)
│   └── reuslt.ipynb
└── src
    ├── coding.py
    ├── main.py
    ├── math_analysis.py
    ├── roleplay.py
    ├── utils.py
    └── writing.py
```


### Data Description

The project provides the evaluation of various tasks and models under the seed value of 2024, with three passes of seed values 2023, 2024, and 2025 in the code generation task. The evaluation includes the responses of each task and model under the default and final system prompts, as well as the assessment of the GPT-3.5-turbo.

- **Model Responses:**
  - `./response/raw_responses_backups/default_prompt`.
  - `./response/raw_responses_backups/final_prompt`.
- **Evaluation of gpt-3.5-turbo:**
  - `./results/gpt_raw_evals_backups/default_prompt`.
  - `./results/gpt_raw_evals_backups/final_prompt`.

In the following section: `TASK` can be `"coding"`, `"math"`, `"roleplay"`, `"writing"`; `MODEL` can be `"qwen-14b-chat"`, `"baichuan2-13b-chat-v1"`, `"gpt-3.5-turbo"`. For detailed project task description, test results and technical reports, please see the [report](report.pdf) (NOTICE: the report is in Chinese.)

**Quickly view the score directly (including the pass@3 metric for code generation tasks):**

```sh
python src/main.py --task TASK --model MODEL --eval --debug
```

Please review the evaluation results for the corresponding tasks and models stored in `./results/gpt_raw_evals`. If you wish to review any previous output records, you can retrieve the corresponding file from the above folder to `./results/gpt_raw_evals` and then run it.

### Code Explanation

The project provides complete code that can be used directly.

- **To request Model Response:** The model will be called via API with the question scope being all the questions for this task in `./data/pj3-test.jsonl`. The response will be saved in `./response/raw_responses`.
  
    ```sh
    python src/main.py --task TASK --model MODEL --request
    # Example:
    python src/main.py --task "math" --model "gpt-3.5-turbo" --request
    ```

- **To view Model Response:** This will output `./response/raw_responses` formatively. Mainly used for debugging, not very meaningful.

    ```sh
    python src/main.py --task TASK --model MODEL --request --debug
    ```

- **To request evaluation for gpt-3.5-turbo (including pass@3 testing if it is a code generation task):**

    ```sh
    python src/main.py --task TASK --model MODEL --eval
    ```

- **Can also run automatically in series:** Note that it is necessary to avoid testing different models at the same time, as it may cause excessive requests for gpt-3.5-turbo (3 Request/min). The code only implements stopping, binary exponential backoff, and other processing for requests of the same model, but does not handle simultaneous testing of different models.

    ```sh
    python src/main.py --task TASK --model MODEL --request --eval
    ```

- **Delete temporary code files:** When the code generation task requests the evaluation of gpt-3.5-turbo, it will generate code files in `./response/code_snippets/MODEL`, create a child process, and run. You can clear it with one click:

    ```sh
    python src/main.py --task TASK --model MODEL --clean_files --clean_response_messages
    ```
