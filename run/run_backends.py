# -*-coding:UTF-8-*-
import argparse
import sys
import os
from scripts.logger.lemon_logger import Logger
import warnings
import datetime
import configparser
from scripts.tools import utils
import ast
import redis
from tensorflow import keras
import tensorflow.keras.backend as K
import pickle
from scripts.tools import utils
import shutil
from itertools import combinations
import numpy as np
from scripts.tools.mutator_selection_logic import Roulette, MCMC
import math
from scripts.tools.utils import DataUtils
from scripts.mutation import model_mutation_generators
import paramiko
import tensorflow as tf

"""Init cuda"""
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""Setting random seed"""

def partially_nan_or_inf(predictions, bk_num):
    """
    Check if there is NAN in the result
    """
    def get_nan_num(nds):
        _nan_num = 0
        for nd in nds:
            if np.isnan(nd).any() or np.isinf(nd).any():
                _nan_num += 1
        return _nan_num

    if len(predictions) == bk_num:
        for input_predict in zip(*predictions):
            nan_num = get_nan_num(input_predict)
            if 0 < nan_num < bk_num:
                return True
            else:
                continue
        return False
    else:
        raise Exception("wrong backend amounts")

def get_selector_by_startegy_name(mutator_s, mutant_s):

    mutant_strategy_dict = {"ROULETTE":Roulette}
    mutator_strategy_dict = {"MCMC":MCMC}
    return mutator_strategy_dict[mutator_s],mutant_strategy_dict[mutant_s]


def save_mutate_history(selector, invalid_history: dict, mutant_history: list):
    mutator_history_path = os.path.join(experiment_dir, "mutator_history.csv")
    mutant_history_path = os.path.join(experiment_dir, "mutant_history.txt")
    with open(mutator_history_path, "w+") as fw:
        fw.write("Name,Success,Invalid,Total\n")
        for op in invalid_history.keys():
            mtrs = selector.mutators[op]
            invalid_cnt = invalid_history[op]
            fw.write("{},{},{},{}\n".format(op, mtrs.delta_bigger_than_zero, invalid_cnt, mtrs.total))
    with open(mutant_history_path, "w+") as fw:
        for mutant in mutant_history:
            fw.write("{}\n".format(mutant))


def is_nan_or_inf(t):
    if math.isnan(t) or math.isinf(t):
        return True
    else:
        return False

def continue_checker(**run_stat):
    start_time = run_stat['start_time']
    time_limitation = run_stat['time_limit']
    cur_counters = run_stat['cur_counters']
    counters_limit = run_stat['counters_limit']
    s_mode = run_stat['stop_mode']

    # if timing
    if s_mode == 'TIMING':
        hours, minutes, seconds = utils.ToolUtils.get_HH_mm_ss(datetime.datetime.now() - start_time)
        total_minutes = hours * 60 + minutes
        mutate_logger.info(f"INFO: Mutation progress: {total_minutes}/{time_limitation} Minutes!")
        if total_minutes < time_limitation:
            return True
        else:
            return False
    # if counters
    elif s_mode == 'COUNTER':
        if cur_counters < counters_limit:
            mutate_logger.info("INFO: Mutation progress {}/{}".format(cur_counters + 1, counters_limit))
            return True
        else:
            return False
    else:
        raise Exception(f"Error! Stop Mode {s_mode} not Found!")


def _generate_and_predict(res_dict, filename, mutate_num, mutate_ops, test_size, exp, backends):
    """
    Generate models using mutate operators and store them
    """
    mutate_op_history = {k: 0 for k in mutate_ops}
    mutate_op_invalid_history = {k: 0 for k in mutate_ops}
    mutant_history = []

    # get mutator selection strategy
    origin_model_name = "{}_origin0.h5".format(exp)
    origin_save_path = os.path.join(mut_dir, origin_model_name)
    mutator_selector_func, mutant_selector_func = get_selector_by_startegy_name(mutator_strategy,mutant_strategy)
    # [origin_model_name] means seed pool only contains initial model at beginning.
    mutator_selector, mutant_selector = mutator_selector_func(mutate_ops), mutant_selector_func([origin_model_name],
                                                                                                capacity=mutate_num + 1)

    shutil.copy(src=filename, dst=origin_save_path)
    origin_model_status, res_dict, accumulative_inconsistency, _ = get_model_prediction(res_dict,
                                                                                        origin_save_path,
                                                                                        origin_model_name, exp,
                                                                                        test_size, backends)
    if not origin_model_status:
        mutate_logger.error(f"Origin model {exp} crashed on some backends! LEMON would skip it")
        sys.exit(-1)

    last_used_mutator = None
    last_inconsistency = accumulative_inconsistency
    mutant_counter = 0

    start_time = datetime.datetime.now()
    order_inconsistency_dict = {}
    run_stat = {'start_time': start_time, 'time_limit': time_limit, 'cur_counters': mutant_counter,
                'counters_limit': mutate_num, 'stop_mode': stop_mode}

    while continue_checker(**run_stat):
        picked_seed = utils.ToolUtils.select_mutant(mutant_selector)
        selected_op = utils.ToolUtils.select_mutator(mutator_selector, last_used_mutator=last_used_mutator)
        mutate_op_history[selected_op] += 1
        last_used_mutator = selected_op
        mutator = mutator_selector.mutators[selected_op]
        mutant = mutant_selector.mutants[picked_seed]

        new_seed_name = "{}-{}{}.h5".format(picked_seed[:-3], selected_op, mutate_op_history[selected_op])
        # seed name would not be duplicate
        if new_seed_name not in mutant_selector.mutants.keys():
            new_seed_path = os.path.join(mut_dir, new_seed_name)
            picked_seed_path = os.path.join(mut_dir, picked_seed)
            mutate_st = datetime.datetime.now()
            # mutate_status = os.system("{}/lemon/bin/python -m  scripts.mutation.model_mutation_generators --model {} "
            #                           "--mutate_op {} --save_path {} --mutate_ratio {}".format(python_prefix,
            #                                                                                    picked_seed_path,
            #                                                                                    selected_op,
            #                                                                                    new_seed_path,
            #                                                                                    flags.mutate_ratio))
            try:
                print("Current {}; Mutate ratio {}".format(selected_op, flags.mutate_ratio))
                origin_model = keras.models.load_model(picked_seed_path, custom_objects=utils.ModelUtils.custom_objects())
                mutated_model = model_mutation_generators.generate_model_by_model_mutation(model=origin_model, operator=selected_op,
                                                                 mutate_ratio=flags.mutate_ratio)

                if mutated_model is None:
                    raise Exception("Error: Model mutation using {} failed".format(selected_op))
                else:
                    mutated_model.save(new_seed_path)

                mutate_et = datetime.datetime.now()
                mutate_dt = mutate_et - mutate_st
                h, m, s = utils.ToolUtils.get_HH_mm_ss(mutate_dt)
                mutate_logger.info("INFO:Mutate Time Used on {} : {}h, {}m, {}s".format(selected_op, h, m, s))

                mutant.selected += 1
                mutator.total += 1
                # execute this model on all platforms
                predict_status, res_dict, accumulative_inconsistency, model_outputs = \
                    get_model_prediction(res_dict, new_seed_path, new_seed_name, exp, test_size, backends)

                if predict_status:
                    mutant_history.append(new_seed_name)

                    if utils.ModelUtils.is_valid_model(inputs_backends=model_outputs, backends_nums=len(backends)):
                        delta = accumulative_inconsistency - last_inconsistency

                        if mutator_strategy == 'MCMC':
                            mutator.delta_bigger_than_zero = mutator.delta_bigger_than_zero + 1 \
                                if delta > 0 else mutator.delta_bigger_than_zero

                        if mutant_strategy == 'ROULETTE' and delta > 0:
                            # when size >= capacity:
                            # random_mutant & Roulette would drop one and add new one
                            if mutant_selector.is_full():
                                mutant_selector.pop_one_mutant()
                            mutant_selector.add_mutant(new_seed_name)
                            last_inconsistency = accumulative_inconsistency

                        mutate_logger.info("SUCCESS:{} pass testing!".format(new_seed_name))
                        mutant_counter += 1
                    else:
                        mutate_op_invalid_history[selected_op] += 1
                        mutate_logger.error("Invalid model Found!")
                else:
                    mutate_logger.error("Crashed or NaN model Found!")
            except Exception as e:
                mutate_logger.error("Exception raised when mutate {} with {}".format(picked_seed, selected_op))
                raise e

            mutate_logger.info("Mutated op used history:")
            mutate_logger.info(mutate_op_history)

            mutate_logger.info("Invalid mutant generated history:")
            mutate_logger.info(mutate_op_invalid_history)

        run_stat['cur_counters'] = mutant_counter

    save_mutate_history(mutator_selector, mutate_op_invalid_history, mutant_history)

    return res_dict


def generate_metrics_result(res_dict, predict_output, model_idntfr):
    mutate_logger.info("Generating Metrics Result")
    accumulative_incons = 0
    backends_pairs_num = 0
    # Compare results pair by pair
    for pair in combinations(predict_output.items(), 2):
        backends_pairs_num += 1
        backend1, backend2 = pair
        bk_name1, prediction1 = backend1
        bk_name2, prediction2 = backend2
        bk_pair = "{}_{}".format(bk_name1, bk_name2)
        for metrics_name, metrics_result_dict in res_dict.items():
            metrics_func = utils.MetricsUtils.get_metrics_by_name(metrics_name)
            # metrics_results in list type
            metrics_results = metrics_func(prediction1, prediction2, y_test[:flags.test_size])

            # ACC -> float: The sum of all inputs under all backends
            accumulative_incons += sum(metrics_results)

            for input_idx, delta in enumerate(metrics_results):
                delta_key = "{}_{}_{}_input{}".format(model_idntfr, bk_name1, bk_name2, input_idx)
                metrics_result_dict[delta_key] = delta

    mutate_logger.info(f"Accumulative Inconsistency: {accumulative_incons}")
    return res_dict, accumulative_incons

def custom_objects():

    def no_activation(x):
        return x

    def leakyrelu(x):
        return K.relu(x, alpha=0.01)

    objects = {}
    objects['no_activation'] = no_activation
    objects['leakyrelu'] = leakyrelu
    return objects

def _get_prediction(bk, x, y, model_path,batch_size,mut_model_name):
    """
    Get prediction of models on different backends
    """
    # test_x, test_y = x[:flags.test_size],y[:flags.test_size]
    test_x, test_y = x[:batch_size],y[:batch_size]
    predict_model = keras.models.load_model(model_path,custom_objects=custom_objects())
    predict_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    main_logger.info("INFO:load model and compile done!")
    res = predict_model.predict(test_x,batch_size=batch_size)
    main_logger.info("SUCCESS:Get prediction for {} successfully on {}!".format(mut_model_name,bk))
    """Store prediction result to redis"""
    redis_conn.hset("prediction_{}".format(mut_model_name),bk,pickle.dumps(res))

def get_model_prediction(res_dict, model_path, model_name, exp, test_size, backends):
    """
    Get model prediction on different backends and calculate distance by metrics
    """
    predict_output = {b: [] for b in backends}
    model_idntfr = model_name[:-3]
    all_backends_predict_status = True

    pool = redis.ConnectionPool(host=lemon_cfg['redis']['host'], port=lemon_cfg['redis']['port'], db=flags.redis_db)
    redis_conn = redis.Redis(connection_pool=pool)
    parameters = lemon_cfg['parameters']
    predict_st = datetime.datetime.now()





    if 'GPU' in device:
        """Init cuda"""
        gpu_ids = parameters['gpu_ids']
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    batch_size = 128
    try:
        """Get model prediction"""
        main_logger.info("INFO:Using {} as backend for states extraction| {} is wanted".format(K.backend(), device))
        x, y = DataUtils.get_data_by_exp(exp_identifier)
        mut_model_name = model_name  ## modified
        _get_prediction(bk=device, x=x, y=y, model_path=model_path, batch_size=batch_size, mut_model_name=mut_model_name)

        """
            Get prediction of models on different backends
            """
        # test_x, test_y = x[:flags.test_size],y[:flags.test_size]
        test_x, test_y = x[:batch_size], y[:batch_size]
        predict_model = keras.models.load_model(model_path, custom_objects=custom_objects())
        # predict_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        main_logger.info("INFO:load model and compile done!")
        res = predict_model.predict(test_x, batch_size=batch_size)
        main_logger.info("SUCCESS:Get prediction for {} successfully on {}!".format(mut_model_name, device))
        """Store prediction result to redis"""
        redis_conn.hset("prediction_{}".format(mut_model_name), device, pickle.dumps(res))

        # If no exception is thrown,save prediction result
        # if pre_status_bk == 0:
        data = pickle.loads(redis_conn.hget("prediction_{}".format(model_name), device))
        predict_output[device] = data

        predict_et = datetime.datetime.now()
        predict_td = predict_et - predict_st
        h, m, s = utils.ToolUtils.get_HH_mm_ss(predict_td)
        mutate_logger.info("Prediction Time Used on {} : {}h, {}m, {}s".format(device, h, m, s))

    except Exception as e:
        import traceback
        traceback.print_exc()
        all_backends_predict_status = False
        mutate_logger.error("{} crash on backend {} when predicting ".format(model_name, device))
        raise e

    status = False
    accumulative_incons = None

    # run ok on all platforms
    if all_backends_predict_status:
        predictions = list(predict_output.values())
        res_dict, accumulative_incons = generate_metrics_result(res_dict=res_dict, predict_output=predict_output, model_idntfr=model_idntfr)

        # If all backends are working fine, check if there exists NAN or INF in the result
        # `accumulative_incons` is nan or inf --> NaN or INF in results
        if is_nan_or_inf(accumulative_incons):
            # has NaN on partial backends
            if partially_nan_or_inf(predictions, len(backends)):
                nan_model_path = os.path.join(nan_dir, f"{model_idntfr}_NaN_bug.h5")
                mutate_logger.error("Error: Found one NaN bug. move NAN model")

            # has NaN on all backends --> not a NaN bug
            else:
                nan_model_path = os.path.join(nan_dir, f"{model_idntfr}_NaN_on_all_backends.h5")
                mutate_logger.error("Error: Found one NaN Model on all libraries. move NAN model")
            shutil.move(model_path, nan_model_path)

        else:  # No NaN or INF on any backend
            mutate_logger.info("Saving prediction")
            with open("{}/prediction_{}.pkl".format(inner_output_dir, model_idntfr), "wb+") as f:
                pickle.dump(predict_output, file=f)
            status = True

    # save crashed model
    else:
        mutate_logger.error("Error: move crash model")
        crash_model_path = os.path.join(crash_dir, model_name)
        shutil.move(model_path, crash_model_path)

    return status, res_dict, accumulative_incons, predict_output

if __name__ == '__main__':

    config_name = sys.argv[1]
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{config_name}")
    parameters = lemon_cfg['parameters']

    flags = argparse.Namespace(
        device=parameters['device'],
        mutate_ops=parameters['mutate_ops'].split(' '),
        exps=parameters['exps'].lstrip().rstrip().split(" "),
        origin_model_dir=parameters['origin_model_dir'],
        output_dir=parameters['output_dir'],
        backend=parameters['backend'].split(' '),
        mutate_num=parameters.getint('mutate_num'),
        mutate_ratio=parameters.getfloat('mutate_ratio'),
        test_size=parameters.getint('test_size'),
        threshold=parameters.getfloat('threshold'),
        redis_db=lemon_cfg['parameters'].getint('redis_db'),
        python_prefix=parameters['python_prefix'].rstrip("/")
    )

    if not os.path.exists(flags.output_dir):
        os.makedirs(flags.output_dir)

    main_logger = Logger()

    # ssh = paramiko.SSHClient()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    # ssh.connect('114.212.191.92', 10022, 'dudapeng', 'ddp_1234', timeout=10)
    # invoke = ssh.invoke_shell()
    # invoke.send("/data/dudapeng/anaconda3/envs/tf115/python /home/dudapeng/workspace//test.py \n")

    # t = paramiko.Transport(('114.212.191.92', 10022))
    # t.connect(username='dudapeng', password='ddp_1234')
    # chan = t.open_session()
    # chan.settimeout(36000)
    # chan.get_pty()
    # chan.invoke_shell()

    """Lemon process"""
    # main_logger.logger.info("Success: Lemon start successfully!")
    start_time = datetime.datetime.now()
    device = flags.device
    # loop model-dataset exp : like lenet5-mnist
    for exp_identifier in flags.exps:

        """Make directory"""
        experiment_dir = os.path.join(flags.output_dir, exp_identifier)
        mut_dir = os.path.join(experiment_dir, "mut_model")
        crash_dir = os.path.join(experiment_dir, "crash")
        nan_dir = os.path.join(experiment_dir, "nan")
        inner_output_dir = os.path.join(experiment_dir, "inner_output")
        metrics_result_dir = os.path.join(experiment_dir, "metrics_result")

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        if not os.path.exists(mut_dir):
            os.makedirs(mut_dir)
        if not os.path.exists(crash_dir):
            os.makedirs(crash_dir)
        if not os.path.exists(nan_dir):
            os.makedirs(nan_dir)
        if not os.path.exists(inner_output_dir):
            os.makedirs(inner_output_dir)
        if not os.path.exists(metrics_result_dir):
            os.makedirs(metrics_result_dir)

        try:
            """Mutate and get output of different backends"""
            main_logger.info("INFO:Lemon mutation starting!")
            main_logger.info("INFO:Lemon for exp: {}".format(exp_identifier))
            origin_model_name = "{}_origin.h5".format(exp_identifier)
            origin_model_file = os.path.join(flags.origin_model_dir,origin_model_name)

            # mutate_lemon = "{}/lemon/bin/python -u -m run.mutate_lemon --mutate_op {} --model {} --output_dir {}" \
            #                 " --backends {} --mutate_num {} --mutate_ratio {} --exp {} --test_size {} --redis_db {} --config_name {}"\
            #                 .format(flags.python_prefix,flags.mutate_ops,origin_model_file,flags.output_dir,flags.backend,
            #                         flags.mutate_num,flags.mutate_ratio,exp_identifier,flags.test_size,flags.redis_db,config_name)
            # os.system(mutate_lemon)

            starttime = datetime.datetime.now()

            time_limit = lemon_cfg['parameters'].getint("time_limit")
            mutator_strategy = lemon_cfg['parameters'].get("mutator_strategy").upper()
            mutant_strategy = lemon_cfg['parameters'].get("mutant_strategy").upper()
            stop_mode = lemon_cfg['parameters'].get("stop_mode").upper()
            alpha = lemon_cfg['parameters'].getfloat("alpha")

            mutate_logger = Logger()
            pool = redis.ConnectionPool(host=lemon_cfg['redis']['host'], port=lemon_cfg['redis']['port'],
                                        db=lemon_cfg['redis'].getint('redis_db'))
            redis_conn = redis.Redis(connection_pool=pool)

            for k in redis_conn.keys():
                if exp_identifier in k.decode("utf-8"):
                    redis_conn.delete(k)

            x, y = utils.DataUtils.get_data_by_exp(exp_identifier)
            x_test, y_test = x[:flags.test_size], y[:flags.test_size]
            pool_size = lemon_cfg['parameters'].getint('pool_size')
            python_prefix = lemon_cfg['parameters']['python_prefix'].rstrip("/")

            try:
                metrics_list = lemon_cfg['parameters']['metrics'].split(" ")
                lemon_results = {k: dict() for k in metrics_list}
                lemon_results = _generate_and_predict(lemon_results, origin_model_file, flags.mutate_num, flags.mutate_ops,
                                                      flags.test_size, exp_identifier, flags.backend)
                with open("{}/{}_lemon_results.pkl".format(experiment_dir, exp_identifier), "wb+") as f:
                    pickle.dump(lemon_results, file=f)
                utils.MetricsUtils.generate_result_by_metrics(metrics_list, lemon_results, metrics_result_dir,
                                                              exp_identifier)

            except Exception as e:
                mutate_logger.exception(sys.exc_info())

            K.clear_session()

            endtime = datetime.datetime.now()
            time_delta = endtime - starttime
            h, m, s = utils.ToolUtils.get_HH_mm_ss(time_delta)
            mutate_logger.info("Mutation process is done: Time used: {} hour,{} min,{} sec".format(h, m, s))

        except Exception:
            main_logger.error("Error: Lemon for exp:{} Failed!".format(exp_identifier))
            main_logger.exception(sys.exc_info())

    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    h,m,s = utils.ToolUtils.get_HH_mm_ss(time_delta)
    main_logger.info("INFO:Lemon is done: Time used: {} hour,{} min,{} sec".format(h,m,s))


