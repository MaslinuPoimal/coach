#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from rl_coach.core_types import ActionType
from rl_coach.filters.action.action_filter import ActionFilter
from rl_coach.spaces import DiscreteActionSpace, ActionSpace
import random


class ActionAdversarialNoise(ActionFilter):
    """
    Apply noise on outputs. For continuous outputs the action will be chosen based on the proximity to the current action.
    """
    def __init__(self):
        """
        :param target_actions: A partial list of actions from the target space to map to.
        :param descriptions: a list of descriptions of each of the actions
        :param covariance: covariance of the applied noise
        """
        # self.target_actions = target_actions
        # self.descriptions = descriptions
        super().__init__()

    def get_unfiltered_action_space(self, output_action_space: ActionSpace) -> DiscreteActionSpace:
        self.output_action_space = output_action_space
        return self.output_action_space

    def filter(self, action: ActionType) -> ActionType:
        return random.randint(self.output_action_space.low, self.output_action_space.high)

    def reverse_filter(self, action: ActionType) -> ActionType:
        # not correct, but since we apply random noise, just pass through
        return action

