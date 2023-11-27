import os
import sys
from convlab.nlu import NLU
from convlab.base_models.llm.base import LLM
from convlab.util import load_dataset, load_ontology, load_database
import json
import re
import random
import configparser
from tqdm import tqdm
from collections import defaultdict
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
      """For example: “1. <DA>[["inform", "hotel", "name": "abc"]]</DA>”. \n"""\
      """Do not generate intents, doamins, slots that are not defined above.\n[/INSTRUCTION]""",
      """[DIALOGUE]\n"""+f"""{{dialogue}}""",
    ])

    return system_instruction

  def predict(self, texts, utterances, domains, example_dialogs):
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
    user_texts = """[USER UTTERANCE]]\n"""+f"""{{user_utterance}}\n"""
    for i, user_text in enumerate(texts):
      user_texts += f'{i+1}. user: {user_text}'
    response = self.model.chat(user_texts)
    self.model.clear_chat_history()
    print(response)
    dialogue_acts = self.normalize_response_to_dialogue_acts(response)
    return dialogue_acts, response

  def normalize_response_to_dialogue_acts(self, response):
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
        if 'user: ' in line or '<DA>' in line:
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
      ut_start_token, ut_end_token = "<UT>", "</UT>"
      da_start_token, da_end_token = "<DA>", "</DA>"
      ut_start_idx = line.find(ut_start_token)
      ut_end_idx = line.find(ut_end_token)
      da_start_idx = line.find(da_start_token)
      da_end_idx = line.find(da_end_token)
      if ut_start_idx == -1 or ut_start_idx == -1:
        user_utterance = 'NO_MATCH_UT'
      else:
        user_utterance = line[ut_start_idx+len(ut_start_token):ut_end_idx].strip()
      if da_start_idx == -1 or da_end_idx == -1:
        dialogue_acts = 'NO_MATCH_DA'
      else:
        dialogue_acts = line[da_start_idx+len(da_start_token):da_end_idx].strip()
      try:
        dialogue_acts = json.loads(dialogue_acts)
        dialogue_acts = [[elm if elm is not None else '' for elm in dialogue_act] for dialogue_act in dialogue_acts]
        join_das = '<das>'.join(['|'.join(da) for da in dialogue_acts])
        normalize_response[no] = {'utter': user_utterance, 'das': join_das}
      except json.decoder.JSONDecodeError:
        normalize_response[no] = {'utter': user_utterance, 'das': dialogue_acts}
      except TypeError:
        normalize_response[no] = {'utter': user_utterance, 'das': dialogue_acts}
    return normalize_response

def get_texts_contexts(dialogue):
  turns = dialogue['turns']
  texts = []
  utterances = []
  for turn in turns:
    if turn['speaker'] == 'user':
      texts.append(turn['utterance'])
    utterances.append(turn['utterance'])
  return texts, utterances

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
  config.read('nlu_config.ini')

  dataset_name = config.get('DATASET', 'name')
  api_type = config.get('API', 'name')
  model_name = config.get('MODEL', 'name')

  dataset = load_dataset(dataset_name)
  fout = f'output/{dataset_name}_{model_name.replace("/", "_")}_nlu_all.json'

  # gpt_model : gpt-3.5-turbo, gpt-4-1106-preview
  nlu = LLM_NLU(dataset_name=dataset_name, api_type=api_type, model_name_or_path=model_name, speaker='user')
  # nlu = LLM_NLU('multiwoz21', 'huggingface', 'Llama-2-7b-chat-hf', 'user', example_dialogs)
  test_datasets = dataset['test']
  print(f'Total test dataset: {len(test_datasets)}')
  for test_data in tqdm(test_datasets):
    texts, utterances = get_texts_contexts(test_data)
    domains = test_data['domains']
    dialogue_id = test_data['dialogue_id']
    example_dialogs = get_example_dialoges(dataset, domains)
    predictions, raw_response = nlu.predict(texts, utterances, domains, example_dialogs)
    # print(predictions)
    dialogue_predictions = {'id': dialogue_id, 'predictions': predictions, 'response': raw_response}
    # print(dialogue_predictions)
    with open(fout, 'a') as f:
      f.write(f'{json.dumps(dialogue_predictions, ensure_ascii=False)}\n')
