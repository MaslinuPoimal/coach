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

from rl_coach.core_types import RewardType
from rl_coach.filters.reward.reward_filter import RewardFilter
from rl_coach.spaces import RewardSpace
import numpy as np


class RewardAdversarialInversionFilter(RewardFilter):
    """
    Inverts the reward with
    """
    def __init__(self, inversion_probility: float):
        """
        :param inversion_probility: The probability with which the sign of the reward will be inverted
        """
        super().__init__()
        self.inversion_probility = min(max(inversion_probility, 0), 0)

    def filter(self, reward: RewardType, update_internal_state: bool=True) -> RewardType:
        reward = float(reward) * np.random.choice([1, -1], p=[1-self.inversion_probility, self.inversion_probility])
        return reward

    def get_filtered_reward_space(self, input_reward_space: RewardSpace) -> RewardSpace:
        input_reward_space.high = max(input_reward_space.high, -input_reward_space.low)
        input_reward_space.low = min(input_reward_space.low, -input_reward_space.high)
        return input_reward_space