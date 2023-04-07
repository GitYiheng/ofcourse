import torch


class Critic(torch.nn.Module):
    def __init__(self, observation_size, hidden_size=256):
        super(Critic, self).__init__()
        self.observation_size = observation_size
        self.action_size = 1  # action is Discrete rather than MultiDiscrete
        self.hidden_size = hidden_size
        self.linear1 = torch.nn.Linear(self.observation_size, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = torch.nn.Linear(self.hidden_size, self.action_size)

    def forward(self, observation):
        output = torch.nn.functional.relu(self.linear1(observation))
        output = torch.nn.functional.relu(self.linear2(output))
        value = self.linear3(output)
        return value


class Actor(torch.nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.linear1 = torch.nn.Linear(self.observation_size, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = torch.nn.Linear(self.hidden_size, self.action_size)

    def forward(self, observation):
        output = torch.nn.functional.relu(self.linear1(observation))
        output = torch.nn.functional.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = torch.distributions.Categorical(torch.nn.functional.softmax(output, dim=-1))
        return distribution
