import json
import os
import re

from openai import OpenAI, completions
import openai


def get_prompts_for_eval(response_root_dir: str, question_prompt_file: str, eval_prompt_file: str, model: str, seed: int, task: str):
    r"""
    Get templatted prompts for evaluation from a file. One prompt for each pass of a question (only "coding" task has multiple passes).

    Return: 
        - list[(question_id, [prompt_for_eval])], Inner list means passes
        - str: sys_prompt
    """
    sys_prompt, prompt_template = get_eval_sys_prompts_and_template(
        eval_prompt_file, task)
    all_question_prompts, all_reference_answers = get_question_prompts(
        question_prompt_file, task)

    # If task == "coding": list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
    # If task != "coding": list[(question_id, [response_message])]. Inner list stands for the turns of the response message.
    all_response_msgs = get_all_response_messages_of_task_from_raw_file(
        response_root_dir, model, seed, task)

    # list[tuple[int, list[str]]], list[(question_id, [prompt_for_eval])], Inner list means passes
    prompts_for_eval = []

    for i in range(len(all_question_prompts)):
        question_id = all_question_prompts[i][0]
        # list[question_msg_of_turns]
        question_prompt_msgs = all_question_prompts[i][1]
        # list[response_msg_of_turns]
        response_msgs = all_response_msgs[i][1]

        if sys_prompt and prompt_template:
            if task == "math":
                prompt_for_eval_this_question = format_eval_prompt(
                    prompt_template, question_prompt_msgs, response_msgs, task, reference_msgs=all_reference_answers[i][1])
            else:
                prompt_for_eval_this_question = format_eval_prompt(
                    prompt_template, question_prompt_msgs, response_msgs, task, reference_msgs=None)
            prompts_for_eval.append(
                (question_id, prompt_for_eval_this_question))
        else:
            print("Error: sys_prompt or prompt_template is None.")

    return prompts_for_eval, sys_prompt


def format_eval_prompt(prompt_template: str, question_msgs: list[str], response_msgs: list[str], task: str, reference_msgs: list[str] = None) -> list[str]:
    r"""
    Format the judge prompt according to the task.

    response_msgs:
        - If task == "coding": list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
        - If task != "coding": list[(question_id, [response_message])]. Inner list stands for the turns of the response message.

    Returns:
        - list[str]: list of formatted judge prompts. Same as above.
    """
    format_args = {}
    ret = []

    if task == "roleplay":
        # multi-turn
        for i, (q, a) in enumerate(zip(question_msgs, response_msgs)):
            format_args[f"question_{i+1}"] = q
            format_args[f"answer_{i+1}"] = a
        ret.append(prompt_template.format(**format_args))
        return ret
    if task == "coding":
        # single-turn, 3 passes
        for i in range(len(response_msgs)):
            format_args["question"] = question_msgs[0]
            format_args["answer"] = response_msgs[i]
            ret.append(prompt_template.format(**format_args))
        return ret
    else:
        # "math", "writing"
        # single-turn
        format_args["question"] = question_msgs[0]
        format_args["answer"] = response_msgs[0]
        if task == "math":
            format_args["ref_answer_1"] = reference_msgs[0]
        ret.append(prompt_template.format(**format_args))
        return ret


def is_chinese(text: str) -> bool:
    r"""
    Detect whether the text contains Chinese characters.
    """
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def extract_avg_rate(gpt_eval_msgs: list[tuple[int, list[str]]]) -> list[tuple[int, float]]:
    r"""
    Extract the average rate of each question from `gpt_eval_msgs`.

    Args:
        - gpt_eval_msgs: list[(question_id, [response_message])]
    
    Returns:
        - list[(question_id, avg_rate)]: list of tuples, each tuple is (question_id, avg_rate)
    """
    avg_rates = []

    for questin_id, eval_msgs in gpt_eval_msgs:
        total_rating = 0
        count = 0
        for msg in eval_msgs:
            # match = re.search(r'Rating: \[\[(\d+(?:\.\d+)?)\]\]', msg)
            match = re.search(r'Rating: \[?\[?(\d+(?:\.\d+)?)\]?\]?', msg)
            if match:
                rating = float(match.group(1))
                total_rating += rating
                count += 1
        if count > 0:
            avg_rate = total_rating / count
            avg_rates.append((questin_id, avg_rate))
        else:
            print("Error: no rating found in question {}".format(questin_id))
    
    return avg_rates



def convert_gpt_completetion_to_dict(gpt_completion) -> dict:
    r"""
    Convert the gpt completion to a dict. Otherwise it cannot be json serialized.
    """
    ret = {
        "id": gpt_completion.id,            # str
        "created": gpt_completion.created,  # int
        "model": gpt_completion.model,      # str
        "system_fingerprint": gpt_completion.system_fingerprint,    # str
        "object": gpt_completion.object,    # str
        "usage": {"prompt_tokens": gpt_completion.usage.prompt_tokens, "completion_tokens": gpt_completion.usage.completion_tokens, "total_tokens": gpt_completion.usage.total_tokens},  # dict
        "choices": []                       # list[dict]
    }

    for choice in gpt_completion.choices:
        ret["choices"].append(
            # str, int, ChatCompletionMesaage
            {"finish_reason": choice.finish_reason, "index": choice.index, "message": {"content": choice.message.content, "role": choice.message.role}})

    return ret


def get_question_prompts(prompt_file: str, task: str) -> tuple[list[tuple[int, list[str]]], list]:
    r"""
    Read the test prompts from jsonl file.

    Returns:
        - list[tuple[int, list[str]]]: [(question_id, list[question_of_different_turns])]
        - list[tuple[int, list[str]]]: [(question_id, list[math_ref_answer])]
    """
    prompts = []
    reference = []
    with open(prompt_file, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get('category', '') == task:
                prompt_id = data.get('question_id', 0)      # int, at least 1
                prompt_content = data.get('turns', [])      # list[str]

                prompts.append((prompt_id, prompt_content))
                if task == "math":
                    ref_content = data.get('reference', [])      # list[str]
                    reference.append((prompt_id, ref_content))

    return prompts, reference


def extract_code(text: str, code_language: str) -> str:
    r"""
    Extract the code snippet from the text. The code snippet is surrounded by ```code_language ```.

    If the start marker is not found, extract everything before the end marker.\
    If the end marker is not found, extract everything after the start marker.\
    If neither is found, return the entire text.\
    If both are found, extract the text between them.
    """

    start_marker = f"```{code_language}"
    end_marker = "```"

    if start_marker in text:
        code_after_start = text.split(start_marker)[1]
        if end_marker in code_after_start:
            code = code_after_start.split(end_marker)[0]
        else:
            code = code_after_start
    else:
        if end_marker in text:
            code = text.split(end_marker)[0]
        else:
            code = text

    return code.strip()


def get_all_response_messages_of_task_from_raw_file(response_root_dir: str, model: str, seed: int, task: str) -> list:
    r"""
    Read the response messages from raw response json file, then return a list of them.

    Raw responses of different tasks should be stored in different files.\
    Specifically, the response comes from `model` of `seed` on `task`.

    Only read the message content, and the semantic meaning of the message differs for different tasks:
        1. coding: the message content is the code snippet
        2. roleplay:
        3. math:
        4. article:

    Args: 
        - response_root_dir: the root directory of all the response files
        - model: model, used to locate the raw response file
        - seed: seed, used to locate the raw response file
        - task: the task of the response file, used to locate the raw response file

    Returns:
        - list[tuple[int, list[str]]]:
            - If task == "coding": list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
            - If task != "coding": list[(question_id, [response_message])]. Inner list stands for the turns of the response message.
    """

    raw_responses_file = os.path.join(
        response_root_dir, "raw_responses", f"{task}_{model}_{seed}.json")

    messages = []

    with open(raw_responses_file, "r") as file:
        data = json.load(file)
        responses_of_all_questions = data.get('question_id', {})
        # responses_of_all_questions: dict.

        for question_id, response_of_a_question in responses_of_all_questions.items():
            # response_of_a_question: list. Responses of a question.
            response_msg_of_a_quetsion = []

            for response_pass in response_of_a_question:
                # response_pass: list. Responses of a pass.
                for response_turn in response_pass:
                    # response_turn: dict. Response of a turn.
                    if model == "gpt-3.5-turbo":
                        choices = response_turn.get('choices', [])
                    else:
                        choices = response_turn.get(
                            'output', {}).get('choices', [])
                    for choice in choices:
                        msg_content = choice.get(
                            'message', {}).get('content', '')
                        if msg_content:
                            msg_content = msg_content.replace(
                                '\\n', '\n').replace('\\r', '\r')
                            response_msg_of_a_quetsion.append(msg_content)
            messages.append((question_id, response_msg_of_a_quetsion))

    # If task == "coding": list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
    # If task != "coding": list[(question_id, [response_message])]. Inner list stands for the turns of the response message.
    return messages


def get_all_eval_messages_of_task_from_raw_file(result_root_dir: str, model: str, seed: int, task: str) -> list:
    r"""
    Read the eval messages from raw result json file, then return a list of them.

    Raw evalution of different tasks should be stored in different files.\
    Specifically, the response comes from `model` of `seed` on `task`.

    Args: 
        - result_root_dir: the root directory of all the result files
        - model: model, used to locate the raw response file
        - seed: seed, used to locate the raw response file
        - task: the task of the response file, used to locate the raw response file

    Returns:
        - list[tuple[int, list[str]]]:
            - If task == "coding": list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
            - If task != "coding": list[(question_id, [response_message])]. Inner list stands for the turns of the response message.
    """

    raw_responses_file = os.path.join(
        result_root_dir, "gpt_raw_evals", f"{task}_{model}_{seed}.json")

    messages = []

    with open(raw_responses_file, "r") as file:
        data = json.load(file)
        responses_of_all_questions = data.get('question_id', {})
        # responses_of_all_questions: dict.

        for question_id, response_of_a_question in responses_of_all_questions.items():
            # response_of_a_question: list. Responses of a question.
            response_msg_of_a_quetsion = []

            for response_pass in response_of_a_question:
                # response_pass: list. Responses of a pass.
                for response_turn in response_pass:
                    # response_turn: dict. Response of a turn.
                    choices = response_turn.get('choices', [])
                    for choice in choices:
                        msg_content = choice.get(
                            'message', {}).get('content', '')
                        if msg_content:
                            msg_content = msg_content.replace(
                                '\\n', '\n').replace('\\r', '\r')
                            response_msg_of_a_quetsion.append(msg_content)
            messages.append((question_id, response_msg_of_a_quetsion))

    # If task == "coding": list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
    # If task != "coding": list[(question_id, [response_message])]. Inner list stands for the turns of the response message.
    return messages


def get_eval_sys_prompts_and_template(judge_prompt_file: str, task: str) -> tuple[str, str]:
    r"""
    Read the judge prompt of the task from the file.

    Return:
        - tuple[str, str]: (sys_prompt, prompt_template)
    """
    target_prompt_name = {"coding": "base-v1",
                          "roleplay": "multi-turn", "math": "math-v1", "writing": "base-v1"}
    with open(judge_prompt_file, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get('name', '') == target_prompt_name.get(task, ''):
                sys_prompt = data.get('system_prompt', '')
                prompt_template = data.get('prompt_template', '')
                return sys_prompt, prompt_template
    print("The judge prompt of task {} does not exist.".format(task))
    return "", ""


def clean_files(task: str, model: str, seed: int, response_root_dir: str, clean_raw_responses=False, clean_response_messages=True):
    r"""
    Delete the response-related files of the given task.

    Args:
        - model: model, used to locate the raw response file
        - seed: seed, used to locate the raw response file
        - task: the task of the response file, used to locate the raw response file
        - response_root_dir: the root directory of all the response files
        - clean_raw_response: whether to delete the raw response file (a json file, in `./responses/raw_responses` folder)
        - clean_response_message: whether to delete the response message file (a json file, in `./responses/response_messages` folder)
    """

    raw_responses_file = os.path.join(
        response_root_dir, "raw_responses", task + "_" + model + "_" + str(seed) + ".json")

    if task == "coding":
        responses_message_dir = os.path.join(
            response_root_dir, "code_snippets", f"{model}_{seed}")

    if clean_raw_responses:
        if os.path.exists(raw_responses_file) and os.path.isfile(raw_responses_file):
            os.remove(raw_responses_file)
            print("The raw response file of task {} has been deleted. Deleted file: {}".format(
                task, raw_responses_file))
        else:
            print("The raw response file of task {} does not exist.".format(task))
    if clean_response_messages:
        if os.path.exists(responses_message_dir):
            if not os.listdir(responses_message_dir):
                print(
                    "The response message file of task {} does not exist.".format(task))
            for file in os.listdir(responses_message_dir):
                file_path = os.path.join(responses_message_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print("The response message file of task {} has been deleted. Deleted file: {}".format(
                        task, file_path))
        else:
            print("The response message file of task {} does not exist.".format(task))
