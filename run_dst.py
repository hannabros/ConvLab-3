from convlab.dst import DST
from convlab.base_models.llm.base import LLM
from convlab.util import load_dataset, load_ontology, load_database

from copy import deepcopy
import os
import sys
import json
import random
import configparser
from tqdm import tqdm
random.seed(1234)


class LLM_DST(DST):
    def __init__(self, dataset_name, api_type, model_name_or_path, generation_kwargs=None):
        self.ontology = load_ontology(dataset_name)
        self.slot_descriptions = None
        self.categorical_slot_values = None
        self.init_system_instruction = self.format_system_instruction(
            self.ontology)
        # print(self.system_instruction)
        self.model = LLM(api_type, model_name_or_path,
                         self.init_system_instruction, generation_kwargs)
        self.state_update = []

    def format_system_instruction(self, ontology):
        # From paper "ChatGPT for Zero-shot Dialogue State Tracking: A Solution or an Opportunity?"
        # http://arxiv.org/abs/2306.01386
        state = ontology['state']
        self.slot_descriptions = deepcopy(ontology['state'])
        self.categorical_slot_values = deepcopy(ontology['state'])

        for domain in state:
            for slot in state[domain]:
                self.slot_descriptions[domain][slot] = ontology['domains'][domain]['slots'][slot]['description']
                if ontology['domains'][domain]['slots'][slot]['is_categorical']:
                    self.categorical_slot_values[domain][slot] = ontology[
                        'domains'][domain]['slots'][slot]['possible_values']
                else:
                    self.categorical_slot_values[domain].pop(slot)
            if self.categorical_slot_values[domain] == {}:
                self.categorical_slot_values.pop(domain)

        system_instruction = "\n\n".join([
            """Consider the following list of concepts , called "slots" provided to you as a json dictionary.""",
            # "\"slots\": "+json.dumps(slot_descriptions, indent=4),
            "\"slots\": "+"{{slot_descriptions}}",
            """Some "slots" can only take a value from predefined list:""",
            # "\"categorical\": "+json.dumps(categorical_slot_values, indent=4),
            "\"categorical\": "+"{{categorical_slot_values}}",
            """Now consider the following dialogue between two parties called the "system" and "user". Can you tell me which of the "slots" were updated by the "user" in its latest response to the "system"?""",
            """Present the updates in **JSON** format, start with <JSON> token and end with </JSON> token. Example: "<JSON>{"hotel": {"name": "abc"}}</JSON>". **Do not forget the "}" token**. If no "slots" were updated, return an empty JSON dictionary. If a user does not seem to care about a discussed "slot" fill it with "dontcare"."""
        ])

        return system_instruction

    def format_turn_prompt(self, user_utterance, system_utterance):
        return '"system": "{}"\n"user": "{}"'.format(system_utterance, user_utterance)

    def normalize_response_to_state_update(self, response):
        start_token, end_token = "<JSON>", "</JSON>"
        start_idx = response.find(start_token)
        end_idx = response.find(end_token)
        if start_idx == -1 or end_idx == -1:
            return {}
        response = response[start_idx+len(start_token):end_idx].strip()
        if response == "":
            return {}
        try:
            state_update = json.loads(response)
        except json.decoder.JSONDecodeError:
            # print('JSONDecodeError')
            # print('*'*30)
            # print([response])
            # print('*'*30)
            return {}
        return state_update

    def update(self, user_action=None):
        assert user_action == None
        context = self.state['history']
        assert len(context) > 0
        if type(context[0]) is list:
            assert len(context[0]) > 1
            context = [item[1] for item in context]
        if len(context) % 2 == 0:
            # system/user/system/user
            assert context[0] == ''
        else:
            # first turn: empty system utterance
            context.insert(0, '')

        assert len(context)//2 >= len(self.state_update) + 1
        for i in range(len(self.state_update), len(context)//2):
            system_utterance = context[2*i]
            user_utterance = context[2*i+1]
            turn_prompt = self.format_turn_prompt(
                user_utterance, system_utterance)
            response = self.model.chat(turn_prompt)
            state_update = self.normalize_response_to_state_update(response)
            # print(turn_prompt)
            # print(response)
            # print(state_update)
            # print('---'*50)
            self.state_update.append(state_update)

        self.state['belief_state'] = deepcopy(self.ontology['state'])
        for state_update in self.state_update:
            for domain in state_update:
                if domain not in self.state['belief_state']:
                    continue
                for slot in state_update[domain]:
                    if slot not in self.state['belief_state'][domain]:
                        continue
                    self.state['belief_state'][domain][slot] = state_update[domain][slot]
        return self.state

    def init_session(self, domains):
        filter_slot_descriptions = {}
        filter_categorical_slot_values = {}
        for domain in domains:
            if domain in self.slot_descriptions:
                filter_slot_descriptions[domain] = self.slot_descriptions[domain]
            if domain in self.categorical_slot_values:
                filter_categorical_slot_values[domain] = self.categorical_slot_values[domain]
        system_instruction = self.init_system_instruction.replace(
            '{{slot_descriptions}}', json.dumps(filter_slot_descriptions, indent=4))
        if filter_categorical_slot_values:
            system_instruction = system_instruction.replace(
                '{{categorical_slot_values}}', json.dumps(filter_categorical_slot_values, indent=4))
        else:
            system_instruction = system_instruction.replace(
                'Some "slots" can only take a value from predefined list:\n\n', "")
            system_instruction = system_instruction.replace(
                "\"categorical\": "+"{{categorical_slot_values}}\n\n", "")
        # print(system_instruction)
        self.state = dict()
        self.state['belief_state'] = deepcopy(self.ontology['state'])
        self.state['booked'] = dict()
        self.state['history'] = []
        self.state['system_action'] = []
        self.state['user_action'] = []
        self.state['terminated'] = False
        self.state_update = []
        self.model.set_system_instruction(system_instruction)
        self.model.clear_chat_history()


def get_texts_contexts(dialogue, depth=None):
    turns = dialogue['turns']
    texts = []
    utterances = []
    contexts = []
    for turn in turns:
        utterances.append(turn['utterance'])
        if turn['speaker'] == 'user':
            texts.append(turn['utterance'])
            contexts.append(utterances[:])
    if depth:
        contexts = [context[-depth::] for context in contexts]
    return texts, contexts


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('dst_config.ini')

    dataset_name = config.get('DATASET', 'name')
    api_type = config.get('API', 'name')
    model_name = config.get('MODEL', 'name')

    dataset = load_dataset(dataset_name)
    fout = f'{dataset_name}_{model_name}_dst.json'

    # gpt_model : gpt-3.5-turbo, gpt-4-1106-preview
    dst = LLM_DST(dataset_name=dataset_name, api_type=api_type,
                  model_name_or_path=model_name)
    # nlu = LLM_NLU('multiwoz21', 'huggingface', 'Llama-2-7b-chat-hf', 'user', example_dialogs)
    test_datasets = dataset['test'][784:]
    print(f'Total test dataset: {len(test_datasets)}')
    for test_data in tqdm(test_datasets):
        texts, contexts = get_texts_contexts(test_data)
        domains = test_data['domains']
        dialogue_id = test_data['dialogue_id']
        dst.init_session(domains)
        predictions = []
        for text, context in zip(texts, contexts):
            # print(text)
            # print(test_data['domains'], example_dialogs, text, context)
            prediction = dst.state['history'] = context
            prediction = dst.update()
            predictions.append({'utter': text, 'pred': prediction})
        dialogue_predictions = {'id': dialogue_id, 'predictions': predictions}
        with open(fout, 'a') as f:
            f.write(f'{json.dumps(dialogue_predictions, ensure_ascii=False)}\n')
