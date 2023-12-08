import re
import json
import copy

class NormalizeNLU():
  def __init__(self, dataset):
    self.dataset = dataset
    self.gold_user_da_list = self.get_gold_user_da()

  def get_gold_user_da(self):
    gold_user_da_list = []
    for data in self.dataset:
      dialogue_id = data['dialogue_id']
      num = 1
      tmp_dict = {}
      for turn in data['turns']:
        if turn['speaker'] == 'user':
          das = turn['dialogue_acts']
          utter = turn['utterance'].strip()
          all_das = []
          for k, v in das.items():
            all_das.extend(v)
          das_conv_form = []
          for da in all_das:
            if da['domain'] == 'general':
              value = ""
            elif 'value' not in da:
              value = ""
            else:
              value = da['value']
            das_conv_form.append([da['intent'], da['domain'], da['slot'], value])
          tmp_dict[num] = {"utter": utter, "das": das_conv_form}
          num += 1
      gold_user_da_list.append({'id': dialogue_id, 'gold': tmp_dict})
    return gold_user_da_list

  def get_gold_user_da_by_id(self, id):
    for da in self.gold_user_da_list:
      if da['id'] == id:
        sorted_dict = dict(sorted(da['gold'].items(), key=lambda item: item[0]))
        return [v['das'] for k, v in sorted_dict.items()]       

  def _get_json_validate_form(self, unvalidate_json, error_message):
    match = re.search(r'\(char (\d+)\)', str(error_message))
    # print('match', match, error_message)
    if match:
      num = int(match.group(1))
      if len(unvalidate_json) == num:
        ### [["bye", "general", ""], 수정
        start_bracket_idx = unvalidate_json.find('[')
        end_bracket_idx = unvalidate_json.rfind(']')
        if start_bracket_idx != -1 and end_bracket_idx != -1:
          unvalidate_json = unvalidate_json[start_bracket_idx+1:end_bracket_idx]
        ### [["request", "hotel", "features", ["free_wifi", "parking"]] 수정
        left_cnt = unvalidate_json.count('[')
        right_cnt = unvalidate_json.count(']')
        if left_cnt < right_cnt:
          fix_json = unvalidate_json + '['
        elif left_cnt > right_cnt:
          fix_json = unvalidate_json + ']'
        result = json.loads(fix_json)
        # print('1', result)
        return result
      ### '[["nobook", "", "", ""]], ["thank", "", "", ""]]' 수정
      if len(unvalidate_json) > num and unvalidate_json[num] == ',':
        if not unvalidate_json.startswith('[['):
          unvalidate_json = '['+unvalidate_json
        if not unvalidate_json.endswith(']]'):
          unvalidate_json = unvalidate_json+']'
        fix_json = unvalidate_json.replace(']],', '],')
        # print('fix_json', fix_json)
        result = json.loads(fix_json)
        # print('2', result)
        return result
      ### [["request", "hotel", "rating": 4, "free_parking": True]] 수정
      ### [["request", "hotel", "features": ["free_wifi", "parking"]] 수정
      if unvalidate_json[num] == ':':
        bracket_pattern = r'\[(.*?)\]'
        no_colon_pattern = r'\"([^\"]+)\"\,'
        colon_pattern = r'\"([^\"]+)\": ([^,\]]+)'
        result = []
        bracket_matches = re.findall(bracket_pattern, unvalidate_json)
        for bracket_match in bracket_matches:
          bracket_match = bracket_match.replace('[', '').replace(']', '')
          no_colon_matches = re.findall(no_colon_pattern, bracket_match)
          # print('no_colon', no_colon_matches)
          colon_matches = re.findall(colon_pattern, bracket_match)
          # print('colon_match', colon_matches)
          for match in colon_matches:
            no_colon_copy = copy.deepcopy(no_colon_matches)
            no_colon_copy.extend([match[0], match[1]])
            result.append(no_colon_copy)
        # print('3', result, bracket_matches, no_colon_matches, colon_matches)
        return result
      print(unvalidate_json)
      return json.loads(unvalidate_json)
    else:
      raise KeyError
      
  def _has_da(self, response):
    # </DA>가 없는 경우, </DA>가 있으나 안의 내용이 없는 경우
    pattern = re.compile(r'<DA>(.*?)</DA>', re.DOTALL)
    matches = re.findall(pattern, response)
    if matches:
      if not all([True if match.strip()!='' else False for match in matches]):
        return False
      else:
        return True
    else:
      return False

  def _get_num_to_da(self, response):
    # 1.와 </DA> 사이 모든 값 추출
    num_pattern = r'(^\d+\.|\n\d+\.)'
    num_matches = re.findall(num_pattern, response)
    matches = []
    if num_matches:
      for num_match in num_matches:
        num_to_da_pattern = re.compile(f'{num_match}(.*?)</DA>', re.DOTALL)
        match = re.search(num_to_da_pattern, response)
        if match:  
          matches.append(match.group(0).strip())
      return matches
    else:
      return False
    
  def _match_to_json(self, num_to_da_match):
    no = int(re.match('\d+',  num_to_da_match).group(0))
    da_pattern = re.compile(r'<DA>(.*?)</DA>', re.DOTALL)
    num_to_da_match = num_to_da_match.replace('[DA]', '<DA>')
    match = re.search(da_pattern, num_to_da_match).group(1)
    match = match.replace(']]>', ']]')
    match = match.replace('])]', ']]')
    match = match.replace('][', '], [')
    match = match.replace('] [', '], [')
    match = match.replace(']], [[', '], [')
    # match = match.replace('": ', '", ')
    match = match.replace('"None"', '""')
    match = match.replace('None', '""')
    match = match.replace('"null"', '""')
    match = match.replace('null', '""')
    match = match.replace('`', '"')
    match = match.replace('“', '"')
    match = match.replace('”', '"')
    match = match.strip()
    try:
      match_json = json.loads(match)
    except json.decoder.JSONDecodeError as json_e:
      # print(match)
      match_json = self._get_json_validate_form(match, json_e)
    return no, match_json

  def get_pred_das(self, id, response):
    # id = pred['id']
    # response = pred['response']
    pred_das = {}
    pred_das['id'] = id
    pred_das['das'] = {}
    gold_user_da = self.get_gold_user_da_by_id(id)
    gold_user_da_cnt = len(gold_user_da)

    if not self._has_da(response):
      return None
    num_to_da_matches = self._get_num_to_da(response)
    if not num_to_da_matches:
      return None
    # print(num_to_da_matches)
    try:
      last_num = int(re.match(r'\d+', num_to_da_matches[-1]).group(0))
      for num_to_da_match in num_to_da_matches:
        no, match_json = self._match_to_json(num_to_da_match)
        pred_das['das'][str(no)] = match_json
    except:
      return None

    # 수집하지 못한 num
    num_not_in_response = [num for num in range(gold_user_da_cnt) if num >= last_num]
    pred_das['num_not_in_response'] = num_not_in_response
    return pred_das

class NormalizeNLG():
  def __init__(self, dataset):
    self.dataset = dataset
    self.gold_sys_response_list = self.get_gold_sys_response()

  def get_gold_sys_response(self):
    gold_sys_response_list = []
    for data in self.dataset:
      dialogue_id = data['dialogue_id']
      num = 1
      tmp_dict = {}
      for turn in data['turns']:
        if turn['speaker'] == 'system':
          utter = turn['utterance'].strip()
          tmp_dict[num] = {"sys_utter": utter}
          num += 1
      gold_sys_response_list.append({"id": dialogue_id, "gold": tmp_dict})
    return gold_sys_response_list

  def get_gold_sys_response_by_id(self, id):
    for sys_rsp in self.gold_sys_response_list:
      if sys_rsp['id'] == id:
        sorted_dict = dict(sorted(sys_rsp['gold'].items(), key=lambda item: item[0]))
        return [v['sys_utter'] for k, v in sorted_dict.items()]
      
  def _has_sys_rsp(self, response):
    # </SUT>가 없는 경우, </SUT>가 있으나 안의 내용이 없는 경우
    sut_pattern = re.compile(r'<SUT>(.*?)</SUT>', re.DOTALL)
    # System: or Sys: 형태
    sys_pattern = re.compile(r'(?:system:|sys:)(.*?[.!?"])$', re.DOTALL | re.MULTILINE)
    sut_matches = re.findall(sut_pattern, response)
    sys_matches = re.findall(sys_pattern, response.lower())
    matches = None
    if sut_matches:
      matches = sut_matches
    if sys_matches:
      matches = sys_matches
    if matches:
      # print(matches)
      if not all([True if match.strip()!='' else False for match in matches]):
        return False
      else:
        return True
    else:
      return False
    
  def _get_num_lines(self, response):
    # 1.와 다음 숫자까지 분리
    is_start = False
    num_lines = []
    tmp = []
    for line in re.split(r'(^\d+\.|\n\d+\.)', response):
      line = line.lstrip()
      if re.match(r'\d+\.', line):
        is_start = True
        if tmp:
          num_lines.append(''.join(tmp))
          tmp = []
      if is_start:
        tmp.append(line)
    if len(tmp) > 0:
      num_lines.append(''.join(tmp))
    return num_lines

  def _match_sut_token(self, response):
    result = []
    for line in response.split('\n'):
      if '<SUT>' in line:
        sut_idx = line.index('<SUT>')+len('<SUT>')
        if '</UUT>' in line:
          prefix = line[:sut_idx]
          suffix = line[sut_idx:]
          suffix = suffix.replace('</UUT>', '</SUT>')
          result.append(prefix+suffix)
          continue
        else:
          result.append(line)
          continue
      elif '</SUT>' in line and line.count('<UUT>') == 1:
        line = line.replace('<UUT>', '<SUT>')
        result.append(line)
        continue     
      else:
        result.append(line)
        continue
    return '\n'.join(result)

  def _get_sys_rsp(self, num_line, is_last):
    # </SUT>가 없는 경우, </SUT>가 있으나 안의 내용이 없는 경우
    sut_pattern = re.compile(r'<SUT>(.*?)</SUT>', re.DOTALL)
    # System: or Sys: 형태
    sys_pattern = re.compile(r'(?:system:|sys:)(.*?[.!?"])$', re.DOTALL | re.MULTILINE)
    no = int(re.match(r'\d+', num_line).group(0))
    if is_last and not sut_pattern:
      for line in num_line.split('\n'):
        line = line.strip()
        sys_match = re.search(sys_pattern, line.lower())
        if sys_match:
          sys_rsp = line[sys_match.start(1):sys_match.end(1)].strip()
          return no, sys_rsp
    else:
      line = num_line.strip()
      sut_match = re.search(sut_pattern, line)
      sys_match = re.search(sys_pattern, line.lower())
      if sut_match:
        sys_rsp = sut_match.group(1)
        return no, sys_rsp
      if sys_match:
        sys_rsp = line[sys_match.start(1):sys_match.end(1)].strip()
        return no, sys_rsp
    return no, None

  # Get system response predictions
  def get_pred_sys_rsp(self, id, response):
    response = re.sub(r'\([^)]*\)', '', response)
    response = response.replace('SUT:', 'System:')
    response = response.replace('[SUT]', '')
    response = response.replace(' \n', '\n')
    # match <SUT> token with </SUT>
    response = self._match_sut_token(response)
    pred_sys_rsp = {}
    pred_sys_rsp['id'] = id
    pred_sys_rsp['sys_rsp'] = {}
    gold_sys_rsp = self.get_gold_sys_response_by_id(id)
    gold_sys_rsp_cnt = len(gold_sys_rsp)

    if not self._has_sys_rsp(response):
      return None
    num_lines = self._get_num_lines(response)
    if len(num_lines) == 0:
      return None
    try:
      num_success = []
      for idx, num_line in enumerate(num_lines):
        if idx == len(num_lines)-1:
          no, sys_rsp = self._get_sys_rsp(num_line, is_last=True)
        else:
          no, sys_rsp = self._get_sys_rsp(num_line, is_last=False)
        if sys_rsp:
          pred_sys_rsp['sys_rsp'][str(no)] = sys_rsp
          num_success.append(no)
      assert len(num_success) > 0
    except:
      return None
    # 수집하지 못한 num
    num_not_in_response = [num for num in range(gold_sys_rsp_cnt) if num > max(num_success)]
    pred_sys_rsp['num_not_in_response'] = num_not_in_response
    return pred_sys_rsp