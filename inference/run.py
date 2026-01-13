import argparse
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from agent import Agent
import math
import inspect
import traceback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=3)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    args = parser.parse_args()

    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    # Validate worker_split
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    llm_model_name = os.environ.get('LLM_MODEL', '')
    if not llm_model_name:
        llm_model_name = os.path.basename(model.rstrip('/'))
    
    dataset_name = os.path.basename(args.dataset.rstrip('/')).replace('.jsonl', '').replace('.json', '')

    # Use LLM_MODEL for directory structure: OUTPUT_PATH/LLM_MODEL/DATASET
    model_dir = os.path.join(output_base, llm_model_name)
    dataset_dir = os.path.join(model_dir, dataset_name)

    print(f"LLM model name: {llm_model_name}")
    print(f"Model dir at: {model_dir}")
    print(f"Creating dataset dir at: {dataset_dir}")

    # Ensure output directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Model name: {llm_model_name}")
    print(f"Data set path: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    data_filepath = f"{args.dataset}"
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
        items = items
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)

    # Split the dataset
    items = items[start_idx:end_idx]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    if total_splits > 1:
        # Add split suffix to output files when using splits
        output_files = {i: os.path.join(dataset_dir, f"iter{i}_split{worker_split}of{total_splits}.jsonl") for i in range(1, roll_out_count + 1)}
    else:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl") for i in range(1, roll_out_count + 1)}

    processed_queries_per_rollout = {}

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass
        processed_queries_per_rollout[rollout_idx] = processed_queries

    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for item in items:
            question = item.get("question", "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                tasks_to_run_all.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'temperature': args.temperature,
                'top_p': args.top_p,
                'presence_penalty': args.presence_penalty
            }
        }

        def create_agent():
            return Agent(
                llm=llm_cfg,
                function_list=["search", "visit", "google_scholar", "CodeExecutor"]
            )

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}
        
        def execute_agent(agent_fn, *fn_args, **fn_kwargs):
            result = agent_fn(*fn_args, **fn_kwargs)
            if inspect.iscoroutine(result):
                return asyncio.run(result)
            return result

        def run_task(task_info):
            agent = create_agent()
            return execute_agent(agent._run, task_info, model)

        if args.max_workers == 1:
            for task_info in tqdm(tasks_to_run_all, desc="Processing All Rollouts"):
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]
                try:
                    result = run_task(task_info)
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except concurrent.futures.TimeoutError:
                    question = task_info["item"].get("question", "")
                    print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": "Timeout (>1800s)",
                        "messages": [],
                        "prediction": "[Failed]"
                    }
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    question = task_info["item"].get("question", "")
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": f"Future resolution failed: {exc}, traceback: {traceback.format_exc()}",
                        "messages": [],
                        "prediction": "[Failed]",
                    }
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        run_task,
                        task
                    ): task for task in tasks_to_run_all
                }

                for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                    task_info = future_to_task[future]
                    rollout_idx = task_info["rollout_idx"]
                    output_file = output_files[rollout_idx]
                    try:
                        result = future.result()
                        with write_locks[rollout_idx]:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    except concurrent.futures.TimeoutError:
                        question = task_info["item"].get("question", "")
                        print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                        future.cancel()
                        error_result = {
                            "question": question,
                            "answer": task_info["item"].get("answer", ""),
                            "rollout_idx": rollout_idx,
                            "rollout_id": rollout_idx,
                            "error": "Timeout (>1800s)",
                            "messages": [],
                            "prediction": "[Failed]"
                        }
                        with write_locks[rollout_idx]:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    except Exception as exc:
                        question = task_info["item"].get("question", "")
                        print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')
                        error_result = {
                            "question": question,
                            "answer": task_info["item"].get("answer", ""),
                            "rollout_idx": rollout_idx,
                            "rollout_id": rollout_idx,
                            "error": f"Future resolution failed: {exc}, traceback: {traceback.format_exc()}",
                            "messages": [],
                            "prediction": "[Failed]",
                        }
                        print("===============================")
                        print(error_result)
                        print("===============================")
                        with write_locks[rollout_idx]:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")