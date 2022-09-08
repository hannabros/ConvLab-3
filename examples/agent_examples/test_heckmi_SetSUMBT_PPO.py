# available NLU models

from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.dst.setsumbt.unified_format_data.Tracker import SetSUMBTTracker
from convlab2.policy.ppo import PPO
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG

from convlab2.dialog_agent import PipelineAgent
from convlab2.util.analysis_tool.analyzer import Analyzer

import random
import numpy as np
from datetime import datetime

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)


def test_end2end(seed=20200202, n_dialogues=1000):

    # Dialogue System, receiving user (simulator) output
    sys_nlu = None
    sys_dst = SetSUMBTTracker(model_type='roberta',
                              model_path="https://cloud.cs.uni-duesseldorf.de/s/Yqkzz8NW3yoMWRk/download/setsumbt_end.zip")
    sys_policy = PPO(False, seed=seed)
    sys_policy.load('/home/heckmi/gcp/supervised')
    sys_nlg = TemplateNLG(is_user=False)
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys', return_semantic_acts=True)

    # User Simulator, receiving system output
    user_nlu = None
    #user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
    #                   model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
    user_dst = None
    user_policy = RulePolicy(character='usr')
    #user_nlg = None
    user_nlg = TemplateNLG(is_user=True)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(seed)
    now = datetime.now()
    time = now.strftime("%Y%m%d%H%M%S")
    name = f'SetSUMBT-PPO-Rule-Seed{seed}-{time}'
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=name, total_dialog=n_dialogues)

if __name__ == '__main__':
    # Get arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='Seed', default=20200202, type=int)
    parser.add_argument('--n_dialogues', help='Number of eval dialogues', default=1000, type=int)
    args = parser.parse_args()

    test_end2end(seed=args.seed, n_dialogues=args.n_dialogues)
