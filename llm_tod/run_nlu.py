import os
import sys
from convlab.nlu import NLU
from convlab.base_models.llm.base import LLM
from convlab.util import load_dataset, load_ontology, load_database
from llm_tod.util.normalize import NormalizeNLU
import json
import re
import random
import configparser
from tqdm import tqdm
from collections import defaultdict
import copy
random.seed(1234)

class LLM_NLU(NLU):
  def __init__(self, dataset_name, api_type, model_name_or_path, speaker, generation_kwargs=None):
    assert speaker in ['user', 'system']
    self.speaker = speaker
    self.opponent = 'system' if speaker == 'user' else 'user'
    self.ontology = load_ontology(dataset_name)
    self.slots = None
    self.initial_system_instruction = self.format_system_instruction(self.ontology)
    self.system_instruction = None
    self.model = LLM(api_type, model_name_or_path, self.system_instruction, generation_kwargs)

  def format_system_instruction(self, ontology):
    intents = {intent: ontology['intents'][intent]['description'] for intent in ontology['intents']}
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
      """[INSTRUCTION]\nYou are an excellent dialogue acts parser. """
      """Dialogue acts are used to represent the intention of the speaker. """
      """Dialogue acts are a list of tuples, each tuple is in the form of (intent, domain, slot, value). """
      """The "intent", "domain", "slot" are defines as follows:""",
      '"intents": '+json.dumps(intents, indent=4),
      '"domain2slots": '+"""{{domain_slot}}""",
      """Here are example dialogue acts:""",
      """{{example}}""",
      """Now consider the [DIALOGUE] and generate the dialogue acts from each [USER UTTERANCE], denoted with numbers. """
      """Each utterance is independent and must not take any other dialogue acts from another utterance. """
      """Start with the same number from [USER UTTERANCE] and each user utterance should place with <UT> token and end with </UT>. """\
      """Right after </UT> token, dialogue acts of each utterance should place with <DA> token and end with </DA> token. \n"""\
      """For example: “1. <DA>[["inform", "hotel", "name", "abc"]]</DA>”. \n"""\
      """Do not generate intents, domains, slots that are not defined above.\n""",
      """Do not generate intents, domains, slots of [DIALOGUE]. Only generate intents, domains, slots of [USER UTTERANCE]\n[/INSTRUCTION]""",
      """[DIALOGUE]\n"""+f"""{{dialogue}}""",
    ])

    return system_instruction

  def predict(self, texts, utterances, domains, example_dialogs, no_to_new_predict=False):
    filter_slots = {}
    for domain in domains:
      filter_slots[domain] = self.slots[domain]
    if dataset_name == 'multiwoz21':
      filter_slots['general'] = {"description": "general domain without slots"}
    # domain specific example
    example = []
    for example_dialog in example_dialogs:
      for i, turn in enumerate(example_dialog['turns']):
        tmp = ""
        if turn['speaker'] == self.speaker:
          if i > 0:
            tmp += example_dialog['turns'][i-1]['speaker'] + \
              ': '+example_dialog['turns'][i-1]['utterance']+'\n'
          tmp += turn['speaker']+': '+turn['utterance']+'\n'
          das = []
          for da_type in turn['dialogue_acts']:
            for da in turn['dialogue_acts'][da_type]:
              intent, domain, slot, value = da.get('intent'), da.get('domain'), da.get('slot', ''), da.get('value', '')
              das.append((intent, domain, slot, value))
          tmp += '<DA>' + json.dumps(das)+'</DA>'
        example.append(tmp)
    examples = '\n\n'.join(example[:5])
    utteracnes_text = ""
    for i, utter in enumerate(utterances):
      if i % 2 == 0:
        utteracnes_text += f'user: {utter}\n'
      else:
        utteracnes_text += f'system: {utter}\n'

    self.system_instruction = self.initial_system_instruction.replace('{{doamin_slot}}', json.dumps(filter_slots, indent=4))
    self.system_instruction = self.system_instruction.replace('{{example}}', examples)
    self.system_instruction = self.system_instruction.replace('{{dialogue}}', utteracnes_text)
    self.model.set_system_instruction(self.system_instruction)
    user_texts = """[USER UTTERANCE]\n"""
    if not no_to_new_predict:
      for i, user_text in enumerate(texts):
        user_texts += f'{i+1}. user: {user_text}\n\n'
    else:
      for i, user_text in zip(no_to_new_predict, texts):
        user_texts += f'{i}. user: {user_text}\n\n'
    response = self.model.chat(user_texts)
    self.model.clear_chat_history()
    return response

def get_texts_contexts(dialogue, no_to_new_predict=False):
  turns = dialogue['turns']
  texts = []
  utterances = []
  for turn in turns:
    if turn['speaker'] == 'user':
      texts.append(turn['utterance'])
    utterances.append(turn['utterance'])
  if not no_to_new_predict:
    return texts, utterances
  else:
    texts_to_new_predict = [text for idx, text in enumerate(texts) if str(idx+1) in no_to_new_predict]
    return texts_to_new_predict, utterances

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
    config_file = './llm_tod/nlu_config.ini'
  config = configparser.ConfigParser()
  config.read(config_file)

  dataset_name = config.get('DATASET', 'name')
  api_type = config.get('API', 'name')
  model_name = config.get('MODEL', 'name')
  clean_result_path = config.get('CLEAN', 'path')
  clean_only = config.getboolean('CLEAN', 'clean_only')

  dataset = load_dataset(dataset_name)
  fout = f'./llm_tod/llm_output/merge/{dataset_name}_{model_name.replace("/", "_")}_nlu_all_merge.json'
  with open(clean_result_path, 'r') as f:
    clean_results = json.load(f)

  # gpt_model : gpt-3.5-turbo, gpt-4-1106-preview
  nlu = LLM_NLU(dataset_name=dataset_name, api_type=api_type, model_name_or_path=model_name, speaker='user')
  # nlu = LLM_NLU('multiwoz21', 'huggingface', 'Llama-2-7b-chat-hf', 'user', example_dialogs)
  if clean_only:
    clean_ids = list(set([result['id'] for result in clean_results]))
    test_datasets = []
    for data in dataset['test']:
      if data['dialogue_id'] in clean_ids:
        test_datasets.append(data)
  else:
    test_datasets = dataset['test']
  normalizer = NormalizeNLU(test_datasets)

  dataset_pred_das = []
  print(f'Total test dataset: {len(test_datasets)}')
  for test_data in tqdm(test_datasets):
    domains = test_data['domains']
    example_dialogs = get_example_dialoges(dataset, domains)
    dialogue_id = test_data['dialogue_id']
    clean_result = get_clean_result_by_id(dialogue_id, clean_results)
    gold_no = len(normalizer.get_gold_user_da_by_id(dialogue_id))
    if clean_result['das'] == 'FAIL':
      clean_result['num_not_in_response'] = [str(i) for i in range(gold_no)]
    pred_cnt = 0 # normalize 완료한 대화 개수
    add_no = [] # 수집해야 할 대화 index
    pred_das_merge = {}
    max_cnt = 0
    while pred_cnt < gold_no and max_cnt < 20:
      if clean_result['das'] == 'FAIL' and len(add_no) == 0:
        texts, utterances = get_texts_contexts(test_data)
        response = nlu.predict(texts, utterances, domains, example_dialogs)
        pred_das = normalizer.get_pred_das(dialogue_id, response)
        if pred_das:
          pred_das_merge = copy.deepcopy(pred_das)
          add_no = pred_das['num_not_in_response']
          pred_cnt += len(pred_das['das'])
      elif len(clean_result['num_not_in_response']) > 0 and len(add_no) == 0:
        no_to_new_predict = clean_result['num_not_in_response']
        no_to_new_predict = [str(int(no)+1) for no in no_to_new_predict]
        texts, utterances = get_texts_contexts(test_data, no_to_new_predict)
        response = nlu.predict(texts, utterances, domains, example_dialogs, no_to_new_predict)
        pred_das = normalizer.get_pred_das(dialogue_id, response)
        if pred_das:
          pred_das_merge = copy.deepcopy(clean_result)
          for k, v in pred_das['das'].items():
            if k not in pred_das_merge['das']:
              pred_das_merge['das'][k] = v
          add_no = [str(no+1) for no in range(gold_no) if str(no+1) not in list(pred_das_merge['das'].keys())]
          pred_cnt = len(pred_das_merge['das'])
      elif len(clean_result['num_not_in_response']) == 0:
        pred_das_merge = clean_result
        pred_cnt = len(clean_result['das'])
      elif len(add_no) > 0:
        texts, utterances = get_texts_contexts(test_data, add_no)
        response = nlu.predict(texts, utterances, domains, example_dialogs, add_no)
        pred_das = normalizer.get_pred_das(dialogue_id, response)
        if pred_das:
          for k, v in pred_das['das'].items():
            if k not in pred_das_merge['das']:
              pred_das_merge['das'][k] = v
          add_no = [str(no+1) for no in range(gold_no) if str(no+1) not in list(pred_das_merge['das'].keys())]
          pred_cnt = len(pred_das_merge['das'])
      max_cnt += 1
      # print(pred_cnt)
    if max_cnt < 20:
      dataset_pred_das.append(pred_das_merge)
      with open(fout, 'a') as f:
        f.write(f'{json.dumps(pred_das_merge, ensure_ascii=False)}\n')

    # with open(fout, 'w') as f:
    #   json.dump(dataset_pred_das, f)
    
    # print(predictions)
    # dialogue_predictions = {'id': dialogue_id, 'predictions': predictions, 'response': raw_response}
    # print(dialogue_predictions)
    # with open(fout, 'a') as f:
    #   f.write(f'{json.dumps(dialogue_predictions, ensure_ascii=False)}\n')
