# Action Space and Observation Space

## Action Space

### Joint Action Space (MultiDiscrete)
```console
>>> env.action_space
[MultiDiscrete([1 3 2 1]), MultiDiscrete([1 2 2 1])]
```

### Sampled Joint Action (MultiDiscrete)
```console
>>> env.action_space.sample()
[array([0, 2, 1, 0]), array([0, 0, 1, 0])]
```

### Joint Action Space (Discrete)
```console
>>> env.action_space_discrete
[Discrete(6), Discrete(4)]
```

### Sampled Joint Action (Discrete)
```console
>>> env.action_space_discrete.sample()
[4, 1]
```

### Sampled Joint Action of Agent 0 (MultiDiscrete)
```console
>>> env.action_space[0].sample()
array([0, 1, 0, 0])
```

### Sampled Joint Action of Agent 1 (MultiDiscrete)
```console
>>> env.action_space[1].sample()
array([0, 1, 0, 0])
```

### Sampled Joint Action of Agent 0 (Discrete)
```console
>>> env.action_space_discrete[0].sample()
5
```

### Sampled Joint Action of Agent 1 (Discrete)
```console
>>> env.action_space_discrete[1].sample()
1
```

### Discrete-MultiDiscrete Joint Action Conversion
```console
>>> joint_action_discrete = 1
>>> env.joint_discrete_to_multi_discrete(joint_action_discrete)
[[0, 0, 0, 0], [0, 0, 1, 0]]
```

### MultiDiscrete-Discrete Joint Action Conversion
```console
>>> joint_action_multi_discrete = [[0, 2, 1, 0], [0, 0, 0, 0]]
>>> env.joint_multi_discrete_to_discrete(joint_action_multi_discrete)
20
```

### Discrete-MultiDiscrete Agent Action Conversion
```console
>>> agent_index = 0
>>> action_discrete = 1
>>> env.agent_discrete_to_multi_discrete(agent_index, action_discrete)
[0, 0, 1, 0]
```

### MultiDiscrete-Discrete Agent Action Conversion
```console
>>> agent_index = 1
>>> action_multi_discrete = [0, 1, 1, 0]
>>> env.agent_multi_discrete_to_discrete(agent_index, action_multi_discrete)
3
```

## Observation Space

### Observation Space Shape of Agent 0
```console
>>> env.observation_space[0].shape[0]
934
```

### Observation Space Shape of Agent 2
```console
>>> env.observation_space[1].shape[0]
932
```
