import os
import sys
from convlab.nlg import NLG
from convlab.base_models.llm.base import LLM
from convlab.util import load_dataset, load_ontology, load_database
import re
import json
import random
import configparser
from collections import defaultdict
from tqdm import tqdm
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
      """Start with the same number from [DIALOGUE ACTS], each user utterance should place with <UUT> token and end with </UUT>. """\
      """Right after </UUT> token, the utterance of system should place with <SUT> token and end with </SUT> token. \n"""\
      """For example: “1. <UUT>user utterance</UUT><SUT>corresponding system utterance by dialogue acts</SUT>". \n"""\
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

  def generate(self, dialogue_acts, texts, example_dialogs, domains):
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
    for i, (dialogue_act, text) in enumerate(zip(dialogue_acts, texts)):
      format_da = self.format_dialogue_acts(dialogue_act)
      prompt += f'{i+1}. '+self.opponent+': '+text+'\n'
      prompt += '<DA>'+json.dumps(format_da)+'</DA>'+'\n\n'
    response = self.model.chat(prompt)
    print(response)
    self.model.clear_chat_history()
    normalize_response = self.normalize_response(response)
    return normalize_response, response

  def normalize_response(self, response):
    split_response = []
    total_line = ""
    for line in response.split('\n'):
      line = line.strip()
      if len(line) == 0:
        continue
      if re.match(r'^\d+\.', line):
        if len(total_line) > 0:
          split_response.append(total_line)
        total_line = line
      else:
        if '<UUT>' in line or '<SUT>' in line:
          total_line += line
    if len(total_line) > 0:
      split_response.append(total_line)
    normalize_response = {}
    no = ""
    for line in split_response:
      if re.match(r'^\d+', line):
        no = re.match(r'^\d+', line).group(0)
      else:
        no = 'None'
      uut_start_token, uut_end_token = "<UUT>", "</UUT>"
      sut_start_token, sut_end_token = "<SUT>", "</SUT>"
      uut_start_idx = line.find(uut_start_token)
      uut_end_idx = line.find(uut_end_token)
      sut_start_idx = line.find(sut_start_token)
      sut_end_idx = line.find(sut_end_token)
      if uut_start_idx == -1 or uut_end_idx == -1:
        user_utterance = 'NO_MATCH_UUT'
      else:
        user_utterance = line[uut_start_idx+len(uut_start_token):uut_end_idx].strip()
      if sut_start_idx == -1 or sut_end_idx == -1:
        system_utterance = 'NO_MATCH_SUT'
      else:
        system_utterance = line[sut_start_idx+len(sut_start_token):sut_end_idx].strip()
      normalize_response[no] = {'user': user_utterance, 'system': system_utterance}
    return normalize_response

def get_das_texts(dialogue):
  turns = dialogue['turns']
  texts = []
  dialogue_acts = []
  for turn in turns:
    if turn['speaker'] == 'user':
      texts.append(turn['utterance'])
    elif turn['speaker'] == 'system':
      dialogue_acts.append(turn['dialogue_acts'])
  return dialogue_acts, texts

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

if __name__ == "__main__":
  config = configparser.ConfigParser()
  config.read('nlg_config.ini')

  dataset_name = config.get('DATASET', 'name')
  api_type = config.get('API', 'name')
  model_name = config.get('MODEL', 'name')

  dataset = load_dataset(dataset_name)
  fout = f"""output/{dataset_name}_{model_name.replace('/', '_')}_nlg_all.json"""

  # gpt_model : gpt-3.5-turbo, gpt-4-1106-preview
  nlg = LLM_NLG(dataset_name=dataset_name, api_type=api_type, model_name_or_path=model_name, speaker='system')
  # nlu = LLM_NLU('multiwoz21', 'huggingface', 'Llama-2-7b-chat-hf', 'user', example_dialogs)
  test_datasets = dataset['test']
  print(f'Total test dataset: {len(test_datasets)}')
  for test_data in tqdm(test_datasets):
    dialogue_acts, texts = get_das_texts(test_data)
    domains = test_data['domains']
    dialogue_id = test_data['dialogue_id']
    example_dialogs = get_example_dialoges(dataset, domains)
    predictions, raw_response = nlg.generate(dialogue_acts, texts, example_dialogs, domains)
    dialogue_predictions = {'id': dialogue_id, 'predictions': predictions, 'raw_response': raw_response}
    with open(fout, 'a') as f:
      f.write(f'{json.dumps(dialogue_predictions, ensure_ascii=False)}\n')
