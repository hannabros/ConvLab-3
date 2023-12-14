import os
import sys
sys.path.append("/home3/hgsun/ConvLab-3")
import copy
from convlab.nlg import NLG
from convlab.base_models.llm.base import LLM
from convlab.util import load_dataset, load_ontology, load_database
import re
import json
import random
import configparser
from collections import defaultdict
from tqdm import tqdm
from llm_tod.util.normalize import NormalizeNLG
random.seed(1234)


class LLM_NLG(NLG):
  def __init__(self, dataset_name, api_type, model_name_or_path, speaker, generation_kwargs=None):
    assert speaker in ['user', 'system']
    self.speaker = speaker
    self.opponent = 'system' if speaker == 'user' else 'user'
    self.ontology = load_ontology(dataset_name)
    self.slots = None
    self.init_system_instruction = self.format_system_instruction(self.ontology)
    # print(self.system_instruction)
    self.model = LLM(api_type, model_name_or_path, self.init_system_instruction, generation_kwargs)

  def format_system_instruction(self, ontology):
    intents = {intent: ontology['intents'][intent]['description'] for intent in ontology['intents']}
    # domains = {domain: '' for domain in ontology['domains']}
    self.slots = defaultdict(dict)
    for domain in ontology['domains']:
      for slot in ontology['domains'][domain]['slots']:
        description = ontology['domains'][domain]['slots'][slot]['description']
        is_categorical = ontology['domains'][domain]['slots'][slot]['is_categorical']
        if 'possible_values' in ontology['domains'][domain]['slots'][slot]:
          possible_values = ontology['domains'][domain]['slots'][slot]['possible_values']
        if possible_values:
          slot_dict = {'description': description, 'is_categorical': is_categorical, 'possible_values': possible_values}
        else:
          slot_dict = {'description': description, 'is_categorical': is_categorical}
        self.slots[domain][slot] = slot_dict

    system_instruction = "\n\n".join([
      """[INSTRUCTION]\nYou are an excellent writing machine. """\
      """You can generate fluent and precise natural language according to the given dialogue acts. """\
      """Dialogue acts are a list of tuples, each tuple is in the form of (intent, domain, slot, value). """\
      """The "intent", "domain", "slot" are defines as follows:""",
      '"intents": '+json.dumps(intents, indent=4),
      '"domain2slots": '+"{{domain_slot}}",
      """Here are some examples:""",
      "{{example}}",
      """Now consider the following [DIALOGUE ACTS]. """\
      """Each dialogue act has previous utterance of user and dialogue act of system based on the user’s utterance. """\
      """Please generate an utterance of system that can express the given dialogue acts precisely. """\
      """Start with the same number from [DIALOGUE ACTS], the utterance of system should place with <SUT> token and end with </SUT> token. \n"""\
      """For example: “1. <SUT>corresponding system utterance by dialogue acts</SUT>". \n"""\
      """Do not generate unrelated intents, domains, and slots that are not in the given dialogue acts.\n[/INSTRUCTION]""",
      """[DIALOGUE ACTS]\n"""
    ])

    return system_instruction

  def format_dialogue_acts(self, dialogue_acts):
    das = []

    if isinstance(dialogue_acts, dict):
      # da in unified format
      for da_type in dialogue_acts:
        for da in dialogue_acts[da_type]:
          intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da.get('value', '')
          das.append((intent, domain, slot, value))
    elif isinstance(dialogue_acts[0], dict):
      # da without da type
      for da in dialogue_acts:
        intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da.get('value', '')
        das.append((intent, domain, slot, value))
    elif isinstance(dialogue_acts[0], list):
      # da is a list of list (convlab-2 format)
      das = dialogue_acts
    else:
      raise ValueError(f"invalid dialog acts format {dialogue_acts}")
    return das

  def generate(self, dialogue_acts, texts, example_dialogs, domains, no_to_new_generate=False):
    filter_slots = {}
    for domain in domains:
      filter_slots[domain] = self.slots[domain]
    if dataset_name == 'multiwoz21':
      filter_slots['general'] = {"description": "general domain without slots"}

    example = []
    for example_dialog in example_dialogs:
      for i, turn in enumerate(example_dialog['turns']):
        tmp = ""
        if turn['speaker'] == self.speaker:
          if i > 0:
            tmp += example_dialog['turns'][i-1]['speaker'] + \
              ': '+example_dialog['turns'][i-1]['utterance']+'\n'
          das = []
          for da_type in turn['dialogue_acts']:
            for da in turn['dialogue_acts'][da_type]:
              intent, domain, slot, value = da.get('intent'), da.get('domain'), da.get('slot', ''), da.get('value', '')
              das.append((intent, domain, slot, value))
          tmp += '<DA>'+json.dumps(das)+'</DA>'+'\n'
          tmp += turn['speaker']+': '+ turn['utterance']
        example.append(tmp)
    examples = '\n\n'.join(example[:5])

    system_instruction = self.init_system_instruction.replace('{{domain_slot}}', json.dumps(filter_slots, indent=4))
    system_instruction = system_instruction.replace('{{example}}', examples)
    self.model.set_system_instruction(system_instruction)
    prompt = ""
    if not no_to_new_generate:
      for i, (dialogue_act, text) in enumerate(zip(dialogue_acts, texts)):
        format_da = self.format_dialogue_acts(dialogue_act)
        prompt += f'{i+1}. '+self.opponent+': '+text+'\n'
        prompt += '<DA>'+json.dumps(format_da)+'</DA>'+'\n\n'
      response = self.model.chat(prompt)
    else:
      for i, dialogue_act, text in zip(no_to_new_generate, dialogue_acts, texts):
        format_da = self.format_dialogue_acts(dialogue_act)
        prompt += f'{i}. '+self.opponent+': '+text+'\n'
        prompt += '<DA>'+json.dumps(format_da)+'</DA>'+'\n\n'
      response = self.model.chat(prompt)
    print(response)
    self.model.clear_chat_history()
    return response

def get_das_texts(dialogue, no_to_new_generate=False):
  turns = dialogue['turns']
  texts = []
  dialogue_acts = []
  for turn in turns:
    if turn['speaker'] == 'user':
      texts.append(turn['utterance'])
    elif turn['speaker'] == 'system':
      dialogue_acts.append(turn['dialogue_acts'])
  if not no_to_new_generate:
    return dialogue_acts, texts
  else:
    dialogue_acts_to_new_generate = [da for idx, da in enumerate(dialogue_acts) if str(idx+1) in no_to_new_generate]
    texts_to_new_generate = [text for idx, text in enumerate(texts) if str(idx+1) in no_to_new_generate]
    return dialogue_acts_to_new_generate, texts_to_new_generate
  
def get_example_dialoges(dataset, domains, cnt=3, turn_threshold=10):
  filter_dataset = []
  example_dialogs = []
  train_dataset = dataset['train']
  for data in train_dataset:
    if len(data['turns']) < turn_threshold:
      filter_dataset.append(data)
  for data in filter_dataset:
    if sorted(data['domains']) == sorted(domains):
      example_dialogs.append(data)
  if len(example_dialogs) == 0:
    for data in filter_dataset:
      if len(set(data['domains']).intersection(domains)) > 0:
        example_dialogs.append(data)
  if len(example_dialogs) < cnt:
    sample = example_dialogs
  else:
    sample = random.sample(example_dialogs, cnt)
  return sample

def get_clean_result_by_id(dialogue_id, clean_results):
  for result in clean_results:
    result_id = result['id']
    if dialogue_id == result_id:
      return result

if __name__ == "__main__":
  if len(sys.argv) == 2:
    config_file = sys.argv[1]
  else:
    config_file = './llm_tod/nlg_config.ini'
  config = configparser.ConfigParser()
  config.read(config_file)

  dataset_name = config.get('DATASET', 'name')
  api_type = config.get('API', 'name')
  model_name = config.get('MODEL', 'name')
  clean_result_path = config.get('CLEAN', 'path')
  clean_only = config.getboolean('CLEAN', 'clean_only')

  dataset = load_dataset(dataset_name)
  fout = f'./llm_tod/llm_output/merge/{dataset_name}_{model_name.replace("/", "_")}_nlg_all_merge.json'
  print(fout)
  with open(clean_result_path, 'r') as f:
    clean_results = json.load(f)

  # gpt_model : gpt-3.5-turbo, gpt-4-1106-preview
  nlg = LLM_NLG(dataset_name=dataset_name, api_type=api_type, model_name_or_path=model_name, speaker='system')
  # nlu = LLM_NLU('multiwoz21', 'huggingface', 'Llama-2-7b-chat-hf', 'user', example_dialogs)
  if clean_only:
    clean_ids = list(set([result['id'] for result in clean_results]))
    test_datasets = []
    for data in dataset['test']:
      if data['dialogue_id'] in clean_ids:
        test_datasets.append(data)
  else:
    test_datasets = dataset['test']
  normalizer = NormalizeNLG(test_datasets)
  dataset_sys_rsp = []
  print(f'Total test dataset: {len(test_datasets)}')
  for test_data in tqdm(test_datasets):
    domains = test_data['domains']
    dialogue_id = test_data['dialogue_id']
    example_dialogs = get_example_dialoges(dataset, domains)
    gold_no = len(normalizer.get_gold_sys_response_by_id(dialogue_id))
    clean_result = get_clean_result_by_id(dialogue_id, clean_results)
    if clean_result['sys_rsp'] == 'FAIL':
      clean_result['num_not_in_response'] = [str(i) for i in range(gold_no)]
    rsp_cnt = 0 # normalize 완료한 대화 개수
    add_no = [] # 수집해야 할 대화 index
    sys_rsp_merge = {}
    max_cnt = 0
    while rsp_cnt < gold_no and max_cnt < 20:
      if clean_result['sys_rsp'] == 'FAIL' and len(add_no) == 0:
        dialogue_acts, texts = get_das_texts(test_data)
        response = nlg.generate(dialogue_acts, texts, example_dialogs, domains)
        pred_sys_rsp = normalizer.get_pred_sys_rsp(dialogue_id, response)
        if pred_sys_rsp:
          sys_rsp_merge = pred_sys_rsp
          sys_rsp_merge['response'] = response
          add_no = [num+1 for num in pred_sys_rsp['num_not_in_response']]
          rsp_cnt += len(pred_sys_rsp['sys_rsp'])
      elif len(clean_result['num_not_in_response']) > 0 and len(add_no) == 0:
        no_to_new_generate = clean_result['num_not_in_response']
        no_to_new_generate = [str(int(no)+1) for no in no_to_new_generate]
        dialogue_acts, texts = get_das_texts(test_data, no_to_new_generate)
        response = nlg.generate(dialogue_acts, texts, example_dialogs, domains, no_to_new_generate)
        pred_sys_rsp = normalizer.get_pred_sys_rsp(dialogue_id, response)
        if pred_sys_rsp:
          sys_rsp_merge = copy.deepcopy(clean_result)
          sys_rsp_merge['response'] = response
          for k, v in pred_sys_rsp['sys_rsp'].items():
            if k not in sys_rsp_merge['sys_rsp']:
              sys_rsp_merge['sys_rsp'][k] = v
          add_no = [str(no+1) for no in range(gold_no) if str(no+1) not in list(sys_rsp_merge['sys_rsp'].keys())]
          rsp_cnt = len(sys_rsp_merge['sys_rsp'])
      elif len(clean_result['num_not_in_response']) == 0:
        sys_rsp_merge = clean_result
        rsp_cnt = len(clean_result['sys_rsp'])
      elif len(add_no) > 0:
        dialogue_acts, texts = get_das_texts(test_data, add_no)
        response = nlg.generate(dialogue_acts, texts, example_dialogs, domains, add_no)
        pred_sys_rsp = normalizer.get_pred_sys_rsp(dialogue_id, response)
        if pred_sys_rsp:
          sys_rsp_merge['response'] += '\n\n'+response
          for k, v in pred_sys_rsp['sys_rsp'].items():
            if k not in sys_rsp_merge['sys_rsp']:
              sys_rsp_merge['sys_rsp'][k] = v
          add_no = [str(no+1) for no in range(gold_no) if str(no+1) not in list(sys_rsp_merge['sys_rsp'].keys())]
          rsp_cnt += len(sys_rsp_merge['sys_rsp'])
      max_cnt += 1
    dataset_sys_rsp.append(sys_rsp_merge)
    if len(sys_rsp_merge) > 0:
      with open(fout, 'a') as f:
        f.write(f'{json.dumps(sys_rsp_merge, ensure_ascii=False)}\n')

    # dialogue_acts, texts = get_das_texts(test_data)
    # predictions, raw_response = nlg.generate(dialogue_acts, texts, example_dialogs, domains)
    # dialogue_predictions = {'id': dialogue_id, 'predictions': predictions, 'raw_response': raw_response}
    # with open(fout, 'a') as f:
    #   f.write(f'{json.dumps(dialogue_predictions, ensure_ascii=False)}\n')
