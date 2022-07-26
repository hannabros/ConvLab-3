import json
from convlab.policy.tus.unify.util import split_slot_name

DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK, ""]


# only support user goal from dataset


class Goal(object):
    """ User Goal Model Class. """

    def __init__(self, goal: list = None):
        """
        create new Goal by random
        Args:
            goal (list): user goal built from user history
            ontology (dict): domains, slots, values
        """
        self.goal = goal
        self.max_domain_len = 6
        self.max_slot_len = 20
        self.local_id = {}

        self.domains = []
        # goal: {domain: {"info": {slot: value}, "reqt": {slot:?}}, ...}
        self.domain_goals = {}
        # status: {domain: {slot: value}}
        self.status = {}
        self.user_history = {}
        self.init_goal_status(goal)
        self.init_local_id()

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'

    def init_goal_status(self, goal):
        for domain, intent, slot, value in goal:  # check this order
            if domain not in self.domains:
                self.domains.append(domain)
                self.domain_goals[domain] = {}

            # "book" domain is not clear for unify data format
            if "info" in intent.lower():
                if "info" not in self.domain_goals[domain]:
                    self.domain_goals[domain]["info"] = {}
                self.domain_goals[domain]["info"][slot] = value

            elif "request" in intent.lower():
                if "reqt" not in self.domain_goals[domain]:
                    self.domain_goals[domain]["reqt"] = {}
                self.domain_goals[domain]["reqt"][slot] = DEF_VAL_UNK

            self.user_history[f"{domain}-{slot}"] = value

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        for domain in self.domain_goals:
            if domain not in self.status:
                return False
            if "info" in self.domain_goals[domain]:
                for slot in self.domain_goals[domain]["info"]:
                    if slot not in self.status[domain]:
                        return False
                    if self.domain_goals[domain]["info"][slot] != self.status[domain][slot]:
                        return False
            if "reqt" in self.domain_goals[domain]:
                for slot in self.domain_goals[domain]["reqt"]:
                    if self.domain_goals[domain]["reqt"][slot] == DEF_VAL_UNK:
                        return False
        return True

    def init_local_id(self):
        # local_id = {
        #     "domain 1": {
        #         "ID": [1, 0, 0],
        #         "SLOT": {
        #             "slot 1": [1, 0, 0],
        #             "slot 2": [0, 1, 0]}}}

        for domain_id, domain in enumerate(self.domains):
            self._init_domain_id(domain)
            self._update_domain_id(domain, domain_id)
            slot_id = 0
            for slot_type in ["info", "book", "reqt"]:
                for slot in self.domain_goals[domain].get(slot_type, {}):
                    self._init_slot_id(domain, slot)
                    self._update_slot_id(domain, slot, slot_id)
                    slot_id += 1

    def insert_local_id(self, new_slot_name):
        # domain, slot = new_slot_name.split('-')
        domain, slot = split_slot_name(new_slot_name)
        if domain not in self.local_id:
            self._init_domain_id(domain)
            domain_id = len(self.domains) + 1
            self._update_domain_id(domain, domain_id)
            self._init_slot_id(domain, slot)
            # the first slot for a new domain
            self._update_slot_id(domain, slot, 0)

        else:
            slot_id = len(self.local_id[domain]["SLOT"]) + 1
            self._init_slot_id(domain, slot)
            self._update_slot_id(domain, slot, slot_id)

    def get_slot_id(self, slot_name):
        # print(slot_name)
        # domain, slot = slot_name.split('-')
        domain, slot = split_slot_name(slot_name)
        if domain in self.local_id and slot in self.local_id[domain]["SLOT"]:
            return self.local_id[domain]["ID"], self.local_id[domain]["SLOT"][slot]
        else:  # a slot not in original user goal
            self.insert_local_id(slot_name)
            domain_id, slot_id = self.get_slot_id(slot_name)
            return domain_id, slot_id

    def action_list(self, sys_act=None):
        priority_action = [x for x in self.user_history]

        if sys_act:
            for _, domain, slot, _ in sys_act:
                slot_name = f"{domain}-{slot}"
                if slot_name and slot_name not in priority_action:
                    priority_action.insert(0, slot_name)

        return priority_action

    def update(self, action: list = None, char: str = "system"):
        # update request and booked
        if char not in ["user", "system"]:
            print(f"unknown role: {char}")
        self._update_status(action, char)
        self._update_goal(action, char)
        return self.status

    def _update_status(self, action: list, char: str):
        for intent, domain, slot, value in action:
            if domain not in self.status:
                self.status[domain] = {}
            # update info
            if "info" in intent:
                self.status[domain][slot] = value
            elif "request" in intent:
                self.status[domain][slot] = DEF_VAL_UNK

    def _update_goal(self, action: list, char: str):
        # update requt slots in goal
        for intent, domain, slot, value in action:
            if "info" not in intent:
                continue
            if self._check_update_request(domain, slot) and value != "?":
                self.domain_goals[domain]['reqt'][slot] = value
                # print(f"update reqt {slot} = {value} from system action")

    def _update_slot(self, domain, slot, value):
        self.domain_goals[domain]['reqt'][slot] = value

    def _check_update_request(self, domain, slot):
        # check whether one slot is a request slot
        if domain not in self.domain_goals:
            return False
        if 'reqt' not in self.domain_goals[domain]:
            return False
        if slot not in self.domain_goals[domain]['reqt']:
            return False
        return True

    def _check_value(self, value=None):
        if not value:
            return False
        if value in NOT_SURE_VALS:
            return False
        return True

    def _init_domain_id(self, domain):
        self.local_id[domain] = {"ID": [0] * self.max_domain_len, "SLOT": {}}

    def _init_slot_id(self, domain, slot):
        self.local_id[domain]["SLOT"][slot] = [0] * self.max_slot_len

    def _update_domain_id(self, domain, domain_id):
        if domain_id < self.max_domain_len:
            self.local_id[domain]["ID"][domain_id] = 1
        else:
            print(
                f"too many doamins: {domain_id} > {self.max_domain_len}")

    def _update_slot_id(self, domain, slot, slot_id):
        if slot_id < self.max_slot_len:
            self.local_id[domain]["SLOT"][slot][slot_id] = 1
        else:
            print(
                f"too many slots, {slot_id} > {self.max_slot_len}")


if __name__ == "__main__":
    data_goal = [["restaurant", "inform", "cuisine", "punjabi"],
                 ["restaurant", "inform", "city", "milpitas"],
                 ["restaurant", "request", "price_range", ""],
                 ["restaurant", "request", "street_address", ""]]
    goal = Goal(data_goal)
    print(goal)
    action = {"char": "system",
              "action": [["request", "restaurant", "cuisine", "?"], ["request", "restaurant", "city", "?"]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
    action = {"char": "user",
              "action": [["inform", "restaurant", "cuisine", "punjabi"], ["inform", "restaurant", "city", "milpitas"]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
    action = {"char": "system",
              "action": [["inform", "restaurant", "price_range", "cheap"]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
    action = {"char": "user",
              "action": [["request", "restaurant", "street_address", ""]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
    action = {"char": "system",
              "action": [["inform", "restaurant", "street_address", "ABCD"]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
