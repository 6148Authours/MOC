import torch.nn as nn


class GoalCurriculum(nn.Module):
    """
    This module is used to generate the goal state for each agent
    Contains:
    1. Gan
    2. meta-gradient for all of the curriculum
    """

    def __init__(self):
        super(GoalCurriculum, self).__init__()

    def forward(self, s_i):
        """

        :param s_i: state s for agent i
        :return: goal state
        """
        pass
