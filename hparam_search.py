"""
    this file includes the codes for tuning hyper-parameters with beam search
"""
import os
import sys
import re
import copy
import time
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser(description='Tuning hyperparameters')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to put logs')
    parser.add_argument('--n_beam', type=int, default=3, help='Beam size')
    parser.add_argument('--highest', action='store_true', help='Keep the highest metrics instead of the lowest')
    parser.add_argument('--gpu', type=int, required=True, help='GPU to use, currently only support one GPU')
    return parser.parse_args()

ARGS = parse()
CACHE_LOG = os.path.join(ARGS.log_dir, 'log_cache.txt')
ALL_LOG = os.path.join(ARGS.log_dir, 'log.txt')
RES_LOG = os.path.join(ARGS.log_dir, 'results.txt')
RES_BUFFER = []


def print_results(s=''):
    global RES_BUFFER
    print(s)
    RES_BUFFER.append(str(s) + '\n')


def write_res_buffer():
    global RES_BUFFER
    with open(RES_LOG, 'a') as fout:
        fout.writelines(RES_BUFFER)
    RES_BUFFER = []


def run(config):

    start = time.time()
    print_results('=' * 20)
    past_values, last_cmd = None, None
    for cmd in cmds():
        cmd_string = cmd.bash_cmd(config, past_values)
        print_results(f'Running {cmd.name} script: {cmd_string}')
        cmd_string += f' > {CACHE_LOG} 2>&1'
        sys.stdout.flush()
        p = os.popen(cmd_string)
        p.close()
        print_results('Finished current script')
        with open(CACHE_LOG, 'r') as fin:
            text = fin.read()
        past_values = cmd.get_values(text)
        last_cmd = cmd
        if cmd.need_record():
            with open(ALL_LOG, 'a') as fout:
                fout.write(text)
    metrics = past_values
    print_results('Results:')
    if isinstance(metrics, dict):
        for key in metrics:
            print_results(f'{key}: {metrics[key]}')
    else:
        print_results(metrics)
    elapsed = time.time() - start
    print_results(f'Elapsed: {round(elapsed, 2)} s')
    return last_cmd.get_score_from_values(metrics)


def beam_search(configs, scores, beam, key, values_choices, highest):
    '''beam search for one parameter'''

    all_res = []  # (config, ppl)-like element
    
    for idx, origin_config in enumerate(configs):
        for value in values_choices:
            config = copy.copy(origin_config)
            if config[key] == value and len(scores) != 0:
                score = scores[idx]
            else:
                config[key] = value
                score = run(config)
            all_res.append((config, score))
            write_res_buffer()
    
    all_res.sort(key=lambda x: x[1], reverse=True if highest else False)
    # return configs and accu
    return [res[0] for res in all_res[:beam]], [res[1] for res in all_res[:beam]]


def main(beam, default_config, hyper_range, seeds, highest):

    print_results(f'The {"higher" if highest else "lower"} metric, the better')

    all_res = []
    train_num = 0
    start = time.time()
    for f in [CACHE_LOG, ALL_LOG, RES_LOG]:
        if os.path.exists(f):
            os.remove(f)
    if not os.path.exists(ARGS.log_dir):
        os.makedirs(ARGS.log_dir)
    for seed in seeds:
        config = copy.copy(default_config)
        config['seed'] = seed
        selected = [copy.copy(config)]
        tries = [(key, hyper_range[key]) for key in hyper_range]
        scores = []
        for key, choices in tries:
            selected, scores = beam_search(selected, scores, beam, key, choices, highest)
            train_num += beam * len(choices)
        print_results('='*20 + f'seed: {seed}' + '='*20)
        print_results(f'top {beam} configs:')
        for cf, a in zip(selected, scores):
            print_results(cf)
            print_results(f'score: {a}')
        all_res.extend([(cur_cf, cur_score) for cur_cf, cur_score in zip(selected, scores)])
    all_res.sort(key=lambda x: x[1], reverse=True)
    all_res = all_res[:beam]
    elapsed = time.time() - start
    print_results()
    print_results("Overall result for beam search:")
    for cf, a in all_res:
        print_results(cf)
        print_results(f'score: {a}')
    print_results(f'Number of tryings: {train_num}')
    print_results(f'Elapsed time: {elapsed} s')
    print_results(f'Average training time: {elapsed / train_num}')
    write_res_buffer()


def config_to_bash_args(config, keys=None):
    cmd = ''
    if keys is None:
        keys = config.keys()
    for key in keys:
        cmd += f'--{key} '
        if config[key] is not None:
            cmd += f'{config[key]} '
    return cmd


class AbstractCMD:
    def __init__(self, name='Anonymous'):
        self.name = name

    ########## Override ##########
    def need_record(self):
        '''
            If return True, all log histories of this command will be recorded in log.txt
        '''
        return True

    ########## Override ##########
    def bash_cmd(self, config, past_values):
        '''
            Use config_to_bash_args(config) to transform config to bash arguments.
            The returned string will be directly run in bash and the outputs will
            be passed to get_values(text)
        '''
        raise NotImplementedError()

    ########## Override ##########
    def get_values(self, text):
        '''
            Extract meaningful values from the outputs of the defined command.
            The returned values will be passed to the next type of command
        '''
        raise NotImplementedError()
    
    ########## Override ##########
    def get_score_from_values(self, values):
        '''
            Get score from extracted values. Must be implemented for the last command
        '''
        raise NotImplementedError()


class TrainCMD(AbstractCMD):
    def need_record(self):
        return False

    def bash_cmd(self, config, past_values):
        cmd = 'TOKENIZERS_PARALLELISM=false python finetune.py '
        keys = [
            'model_type', 'config_path', 'train_set',
            'dev_set', 'save_dir', 'lr', 'max_epochs',
            'gpus', 'warm_up', 'weight_decay', 'batch_size',
            'max_len', 'load_pretrain', 'model_trick',
            'metric', 'alpha', 'fgm', 'fgm_eps', 'seed'
        ]
        return cmd + config_to_bash_args(config, keys)
    
    def get_values(self, text):
        res = re.findall(r'lightning_logs/version_(\d+)/checkpoints/epoch=(\d+)-step=(\d+).ckpt', text)
        return res[-1]  # (version, epoch, step)


class InferCMD(AbstractCMD):
    def need_record(self):
        return False
    
    def bash_cmd(self, config, past_values):
        cmd = 'TOKENIZERS_PARALLELISM=false python infer.py '
        keys = [
            'config_path', 'test_set', 'gpus', 'batch_size', 'max_len'
        ]
        cmd = cmd + config_to_bash_args(config, keys)
        version, epoch, step = past_values
        ckpt_dir = f'{config["save_dir"]}/lightning_logs/version_{version}'
        ckpt_configs = {
            'output': f'{ckpt_dir}/infer.txt',
            'ckpt': f'{ckpt_dir}/checkpoints/epoch={epoch}-step={step}.ckpt'
        }
        cmd += config_to_bash_args(ckpt_configs)
        self.infer_file = ckpt_configs['output']
        return cmd

    def get_values(self, text):
        return self.infer_file


class EvalCMD(AbstractCMD):
    def bash_cmd(self, config, past_values):
        cmd = 'python tools/evaluate.py '
        eval_config = {
            'prediction': past_values,
            'reference': config['test_set']
        }
        cmd += config_to_bash_args(eval_config)
        return cmd

    def get_values(self, text):
        res = re.search(r'\'F1\': ([0-9]*\.?[0-9]+)', text).group(1)
        return float(res)

    def get_score_from_values(self, values):
        return values


########## Override ##########
def cmds():
    '''
        define the command sequence
        the names will be printed, and default is anonymous
    '''
    return [TrainCMD('train'), InferCMD('infer'), EvalCMD('eval')]


if __name__ == '__main__':

    repo_prefix = '/data/private/kxz/GAIIC_2022'
    ner_prefix = f'{repo_prefix}/ner'
    model_type = 'chinese-roberta-wwm-ext'
    default_config = {
        # for training
        'model_type': model_type,
        'config_path': f'{repo_prefix}/checkpoints/{model_type}',
        'train_set': f'{ner_prefix}/data/label_train_data/train.json',
        'dev_set': f'{ner_prefix}/data/label_train_data/dev.json',
        'save_dir': f'{ner_prefix}/results/{model_type}',
        'lr': 2e-5,
        'max_epochs': 5,
        'gpus': ARGS.gpu,
        'warm_up': 0.01,
        'weight_decay': 0.1,
        'batch_size': 32,
        'max_len': 256,
        'load_pretrain': f'{ner_prefix}/results/{model_type}/all_data_pretrain/epoch=61-step=239319.ckpt',
        'model_trick': 'egp',
        'metric': 'f1',
        'alpha': 0.2,
        'fgm': None,
        'fgm_eps': 0.1,
        # for inference
        'test_set': f'{ner_prefix}/data/label_train_data/dev.json',
        # utils
        
    }

    # seeds to try (it will be added to the config dict)
    seeds = [2022, 42, 37587]

    # hyper-parameter choices
    # since learning rate and epochs are strongly correlated, we recommand
    # optimize them in adjacent order
    hyper_range = {
        'lr': [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
        'max_epochs': [4, 5, 6],
        'fgm_eps': [0.1, 1, 3, 5],
        'warm_up': [0.01, 0.03, 0.06],
        'weight_decay': [0.1, 0.01]
    }

    main(beam=ARGS.n_beam, default_config=default_config, hyper_range=hyper_range, seeds=seeds, highest=ARGS.highest)
