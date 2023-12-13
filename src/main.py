import argparse
from ast import arg
import os
from http import HTTPStatus

import utils
import coding
import roleplay
import math_analysis
import writing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="eval mode, will not do any request to the model (except eval via gpt-3.5-turbo). turn off this flag to request the model.")
    parser.add_argument("--request", action="store_true",
                        help="request mode, will not do any evaluation. turn off this flag to do evaluation.")

    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen-14b-chat", "baichuan2-13b-chat-v1", "gpt-3.5-turbo"], help="tested model name.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["coding", "math", "roleplay", "writing"], help="task name.")
    parser.add_argument("--seed", type=int,
                        default=2024, help="seed of all requests.")
    parser.add_argument("--question_prompt_file", type=str,
                        default="./data/question-prompts.jsonl", help="question prompts file")
    parser.add_argument("--eval_prompt_file", type=str,
                        default="./data/judge-prompts.jsonl", help="eval (judge) prompts file")
    parser.add_argument("--response_root_dir", type=str,
                        default="./response", help="response root dir")
    parser.add_argument("--eval_root_dir", type=str,
                        default="./result", help="eval root dir")

    parser.add_argument("--debug", action="store_true",
                        help="debug mode, wil print additional information.")

    parser.add_argument("--clean_files", action="store_true",
                        help="clean files of the given task.")
    parser.add_argument("--clean_raw_responses", action="store_true",
                        help="clean raw responses of the given task.")
    parser.add_argument("--clean_response_messages", action="store_true",
                        help="clean response messages of the given task.")

    args = parser.parse_args()
    model = args.model
    task = args.task
    seed = args.seed
    question_prompt_file = args.question_prompt_file
    eval_prompt_file = args.eval_prompt_file
    response_root_dir = args.response_root_dir
    eval_root_dir = args.eval_root_dir
    raw_response_root_dir = os.path.join(response_root_dir, "raw_responses")
    code_snippets_storage_root_dir = os.path.join(
        response_root_dir, "code_snippets")
    raw_eval_dir = os.path.join(eval_root_dir, "gpt_raw_evals")

    # clean files
    if args.clean_files:
        utils.clean_files(task, model, seed, response_root_dir, clean_raw_responses=args.clean_raw_responses,
                          clean_response_messages=args.clean_response_messages)
        return

    # user question prompts: list[tuple[int, list[str]]], [(question_id, list[question_of_different_turns])]
    # reference_answers: only used in task "math". list[tuple[int, list[str]]], [(question_id, list[question_of_different_turns])]
    question_prompts, reference_answers = utils.get_question_prompts(
        question_prompt_file, task)

    if args.debug:
        print("===============================")
        print("question_prompts: ", question_prompts)
        print("===============================")
        print("reference_answers: ", reference_answers)

    if args.task == "coding":

        # # TODO: Clean this test
        # question_prompts = [(1, ["实现一个Python程序，逐行读取文本文件并计算文件中特定单词的出现次数。"])]
        # reference_answers = []

        """ Request mode """

        if args.request:
            # write raw responses to a json file
            print("===============================")
            print("Waiting for response from the model")
            # only really request if not in debug mode
            if args.debug:
                print("Debug mode. Will not request the model. Last request have been saved in {}/raw_responses/{}_{}_{}.json".format(
                    raw_response_root_dir, task, model, seed))
            else:
                coding.get_raw_response_of_all_questions_of_task(
                    model, question_prompts, raw_response_root_dir, seed)

        """ Will be run in both request and eval mode """
        # list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
        code_snippets_with_descriptions = utils.get_all_response_messages_of_task_from_raw_file(
            response_root_dir, model, seed, task)
        code_snippets = []
        for i in range(len(code_snippets_with_descriptions)):
            code_snippets.append((code_snippets_with_descriptions[i][0], []))
            for j in range(len(code_snippets_with_descriptions[i][1])):
                code_snippets[i][1].append(utils.extract_code(
                    code_snippets_with_descriptions[i][1][j], "python"))

        print("===============================")
        print("===============================")
        print("Requesting done")

        if args.debug:
            print("===============================")
            print("code_snippets_with_descriptions: ",
                  code_snippets_with_descriptions)
            print("===============================")
            print("code_snippets: ", code_snippets)

        """ Eval mode """

        if args.eval:
            # create sub-processes and run python files, to evaluate pass@3 metrics
            print("===============================")
            print("Evaluating pass@3 metrics")
            # list[(questin_id, pass_3_success]
            pass_3_result = coding.eval_pass_3_of_all_questions_of_coding(
                code_snippets, code_snippets_storage_root_dir, model, seed, verbose_result=args.debug)
            print("===============================")
            print("pass_3_result: ", pass_3_result)

            # evaluate via gpt-3.5-turbo
            print("===============================")
            print("Evaluating via gpt-3.5-turbo")
            # list[(question_id, [eval_response])]
            prompts_for_eval, sys_prompt = utils.get_prompts_for_eval(
                response_root_dir, question_prompt_file, eval_prompt_file, model, seed, task)

            # if args.debug:
            #     print("===============================")
            #     print("1st element of prompts_for_eval: ", prompts_for_eval[0])

            if args.debug:
                print("===============================")
                print(
                    "Debug mode. Will not request the model. Last request have been saved in {raw_eval_dir}/{task}_{model}_{seed}.json", raw_eval_dir, task, model, seed)
            else:
                coding.get_gpt_raw_eval_of_all_questions_of_task(
                    sys_prompt, model, prompts_for_eval, raw_eval_dir, seed)

            # list[(question_id, [response_message])]. Inner list stands for the 3 passes.
            gpt_eval_msgs = utils.get_all_eval_messages_of_task_from_raw_file(
                eval_root_dir, model, seed, task)
            gpt_eval_msgs = utils.extract_avg_rate(gpt_eval_msgs)
            if args.debug:
                print("===============================")
                print("gpt_eval_msgs: ", gpt_eval_msgs)

            print("===============================")
            print("===============================")
            print("Evaluating done")
            print("pass@3 metrics: ", pass_3_result)
            print("gpt-3.5-turbo eval: ", gpt_eval_msgs)



    if args.task == "roleplay":

        # # TODO: Clean this test
        # question_prompts = [(1, ["实现一个Python程序，逐行读取文本文件并计算文件中特定单词的出现次数。"])]
        # reference_answers = []

        """ Request mode """

        if args.request:
            # write raw responses to a json file
            print("===============================")
            print("Waiting for response from the model")
            # only really request if not in debug mode
            if args.debug:
                print("Debug mode. Will not request the model. Last request have been saved in {}/raw_responses/{}_{}_{}.json".format(
                    raw_response_root_dir, task, model, seed))
            else:
                roleplay.get_raw_response_of_all_questions_of_task(
                    model, question_prompts, raw_response_root_dir, seed)

        """ Will be run in both request and eval mode """
        # list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
        roleplay_responses = utils.get_all_response_messages_of_task_from_raw_file(
            response_root_dir, model, seed, task)

        print("===============================")
        print("===============================")
        print("Requesting done")

        if args.debug:
            print("===============================")
            print("roleplay_responses: ", roleplay_responses)

        """ Eval mode """

        if args.eval:
            # evaluate via gpt-3.5-turbo
            print("===============================")
            print("Evaluating via gpt-3.5-turbo")
            # list[(question_id, [eval_response])]
            prompts_for_eval, sys_prompt = utils.get_prompts_for_eval(
                response_root_dir, question_prompt_file, eval_prompt_file, model, seed, task)

            # if args.debug:
            #     print("===============================")
            #     print("1st element of prompts_for_eval: ", prompts_for_eval[0])

            if args.debug:
                print("===============================")
                print(
                    "Debug mode. Will not request the model. Last request have been saved in {raw_eval_dir}/{task}_{model}_{seed}.json", raw_eval_dir, task, model, seed)
            else:
                roleplay.get_gpt_raw_eval_of_all_questions_of_task(
                    sys_prompt, model, prompts_for_eval, raw_eval_dir, seed)

            # list[(question_id, [response_message])]. Inner list stands for the 3 passes.
            gpt_eval_msgs = utils.get_all_eval_messages_of_task_from_raw_file(
                eval_root_dir, model, seed, task)
            gpt_eval_msgs = utils.extract_avg_rate(gpt_eval_msgs)
            if args.debug:
                print("===============================")
                print("gpt_eval_msgs: ", gpt_eval_msgs)

            print("===============================")
            print("===============================")
            print("Evaluating done")
            print("gpt-3.5-turbo eval: ", gpt_eval_msgs)



    if args.task == "writing":


        """ Request mode """

        if args.request:
            # write raw responses to a json file
            print("===============================")
            print("Waiting for response from the model")
            # only really request if not in debug mode
            if args.debug:
                print("Debug mode. Will not request the model. Last request have been saved in {}/raw_responses/{}_{}_{}.json".format(
                    raw_response_root_dir, task, model, seed))
            else:
                writing.get_raw_response_of_all_questions_of_task(
                    model, question_prompts, raw_response_root_dir, seed)

        """ Will be run in both request and eval mode """
        # list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
        writing_responses = utils.get_all_response_messages_of_task_from_raw_file(
            response_root_dir, model, seed, task)

        print("===============================")
        print("===============================")
        print("Requesting done")

        if args.debug:
            print("===============================")
            print("writing_responses: ", writing_responses)

        """ Eval mode """

        if args.eval:
            # evaluate via gpt-3.5-turbo
            print("===============================")
            print("Evaluating via gpt-3.5-turbo")
            # list[(question_id, [eval_response])]
            prompts_for_eval, sys_prompt = utils.get_prompts_for_eval(
                response_root_dir, question_prompt_file, eval_prompt_file, model, seed, task)

            # if args.debug:
            #     print("===============================")
            #     print("1st element of prompts_for_eval: ", prompts_for_eval[0])

            if args.debug:
                print("===============================")
                print(
                    "Debug mode. Will not request the model. Last request have been saved in {raw_eval_dir}/{task}_{model}_{seed}.json", raw_eval_dir, task, model, seed)
            else:
                writing.get_gpt_raw_eval_of_all_questions_of_task(
                    sys_prompt, model, prompts_for_eval, raw_eval_dir, seed)

            # list[(question_id, [response_message])]. Inner list stands for the 3 passes.
            gpt_eval_msgs = utils.get_all_eval_messages_of_task_from_raw_file(
                eval_root_dir, model, seed, task)
            gpt_eval_msgs = utils.extract_avg_rate(gpt_eval_msgs)
            if args.debug:
                print("===============================")
                print("gpt_eval_msgs: ", gpt_eval_msgs)

            print("===============================")
            print("===============================")
            print("Evaluating done")
            print("gpt-3.5-turbo eval: ", gpt_eval_msgs)



    if args.task == "math":


        """ Request mode """

        if args.request:
            # write raw responses to a json file
            print("===============================")
            print("Waiting for response from the model")
            # only really request if not in debug mode
            if args.debug:
                print("Debug mode. Will not request the model. Last request have been saved in {}/raw_responses/{}_{}_{}.json".format(raw_response_root_dir, task, model, seed))
            else:
                math_analysis.get_raw_response_of_all_questions_of_task(model, question_prompts, raw_response_root_dir, seed)
        
        """ Will be run in both request and eval mode """
        # list[(question_id, [3_code_snippets])]. Inner list stands for the 3 passes.
        math_responses = utils.get_all_response_messages_of_task_from_raw_file(response_root_dir, model, seed, task)
        
        print("===============================")
        print("===============================")
        print("Requesting done")

    
        if args.debug:
            print("===============================")
            print("math_responses: ", math_responses)

        """ Eval mode """
        
        if args.eval:
            # evaluate via gpt-3.5-turbo
            print("===============================")
            print("Evaluating via gpt-3.5-turbo")
            # list[(question_id, [eval_response])]
            prompts_for_eval, sys_prompt = utils.get_prompts_for_eval(response_root_dir, question_prompt_file, eval_prompt_file, model, seed, task)

            # if args.debug:
            #     print("===============================")
            #     print("1st element of prompts_for_eval: ", prompts_for_eval[0])
            
            if args.debug:
                print("===============================")
                print("Debug mode. Will not request the model. Last request have been saved in {raw_eval_dir}/{task}_{model}_{seed}.json", raw_eval_dir, task, model, seed)
            else:
                math_analysis.get_gpt_raw_eval_of_all_questions_of_task(sys_prompt, model, prompts_for_eval, raw_eval_dir, seed)

            # list[(question_id, [response_message])]. Inner list stands for the 3 passes.
            gpt_eval_msgs = utils.get_all_eval_messages_of_task_from_raw_file(eval_root_dir, model, seed, task)
            gpt_eval_msgs = utils.extract_avg_rate(gpt_eval_msgs)
            if args.debug:
                print("===============================")
                print("gpt_eval_msgs: ", gpt_eval_msgs)

            print("===============================")
            print("===============================")
            print("Evaluating done")
            print("gpt-3.5-turbo eval: ", gpt_eval_msgs)

if __name__ == "__main__":
    main()
