from functools import partial
import multiprocessing
import time
import psutil
import traceback
from pathos.multiprocessing import ProcessingPool as Pool


def process_mode(task, params_dict_list=None, multiy_process_params=[0],
                 batch_size=psutil.cpu_count(logical=False) - 2,
                 is_multi_processing=False):
    assert params_dict_list is not None and len(params_dict_list) == len(multiy_process_params)

    if is_multi_processing:
        task_args = list(zip(multiy_process_params, params_dict_list))
        result_file_list = batch_process(task, task_args, batch_size)
    else:
        result_file_list = []
        for i, multiy_process_param in enumerate(multiy_process_params):
            try:
                result_file = task(multiy_process_param, params_dict_list[i])
                result_file_list.append({
                    "process_id": multiy_process_param,
                    "success": True,
                    "result": result_file,
                    "error": None,
                })
            except Exception:
                result_file_list.append({
                    "process_id": multiy_process_param,
                    "success": False,
                    "result": None,
                    "error": traceback.format_exc(),
                })

    return result_file_list


def safe_task_wrapper(args):
    task, process_param, params_dict = args
    try:
        result = task(process_param, params_dict)
        return {
            "process_id": process_param,
            "success": True,
            "result": result,
            "error": None,
        }
    except Exception:
        return {
            "process_id": process_param,
            "success": False,
            "result": None,
            "error": traceback.format_exc(),
        }


def batch_process(task, task_args, batch_size):
    result_file_list = []

    for i in range(0, len(task_args), batch_size):
        batch = task_args[i: i + batch_size]

        with Pool(processes=len(batch)) as pool:
            wrapped_args = [(task, process_param, params_dict) for process_param, params_dict in batch]
            batch_results = pool.map(safe_task_wrapper, wrapped_args)

        result_file_list.extend(batch_results)

        # 停顿一段时间，避免 CPU 占用过高
        time.sleep(10)

    return result_file_list