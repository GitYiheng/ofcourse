import itertools
import time
from ast import literal_eval

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.container import Buffer, Inventory
from env.utils import MultiAgentActionSpace, MultiAgentObservationSpace
from env.visualizer_client import VisualizerClient


class BaseFulfillmentEnv(gym.Env):
    def __init__(self, args, get_agents):
        self.args = args
        self._get_agents = get_agents
        self.agents = self._get_agents()

        self.n_agents = len(self.agents)
        self.max_step = 200
        self.step_count = None
        self.agent_dones = [False for _ in range(self.n_agents)]
        self.undelivered_penalty = self.args.undelivered_penalty
        self.visualizer = (
            VisualizerClient(getattr(self.args, "visualizer_url", "http://localhost:5000/api/state"))
            if getattr(self.args, "visualize", False)
            else None
        )

        self.observation_space = MultiAgentObservationSpace([
            spaces.Box(
                low=np.zeros(len(agent.get_obs()), dtype=np.float32),
                high=np.full(len(agent.get_obs()), fill_value=np.finfo(np.float32).max, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.agents
        ])
        self.action_space = MultiAgentActionSpace([
            spaces.MultiDiscrete([len(unit.operations) for unit in agent.fulfillment_units])
            for agent in self.agents
        ])
        self._init_action_space()
        self._init_joint_action_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agents = self._get_agents()
        self.step_count = 0
        self.agent_dones = [False for _ in range(self.n_agents)]
        self._send_visualization_state(rewards=None, actions=None)
        return [self.get_agent_obs(agent_i) for agent_i in range(self.n_agents)], {}

    def step(self, agents_action):
        self.step_count += 1
        costs = [0 for _ in range(self.n_agents)]
        for agent_i, action in enumerate(agents_action):
            if not self.agent_dones[agent_i]:
                costs[agent_i] = self.agents[agent_i].step(action, self.step_count)

        truncated = [False for _ in range(self.n_agents)]
        if self.step_count >= self.max_step:
            for agent_i in range(self.n_agents):
                costs[agent_i] += self.agents[agent_i].clearance(self.undelivered_penalty)
                truncated[agent_i] = True
                self.agent_dones[agent_i] = True

        observations = [self.get_agent_obs(agent_i) for agent_i in range(self.n_agents)]
        rewards = [-cost + self.args.r_baseline for cost in costs]
        terminated = [False for _ in range(self.n_agents)]
        self._send_visualization_state(rewards=rewards, actions=agents_action)
        return observations, rewards, terminated, truncated, {}

    def _send_visualization_state(self, rewards=None, actions=None):
        if self.visualizer is None:
            return
        self.visualizer.send(self.build_visualization_state(rewards=rewards, actions=actions))

    def build_visualization_state(self, rewards=None, actions=None):
        agents = []
        orders = []
        total_buffered = 0
        total_inventory = 0
        total_capacity = 0

        for agent_i, agent in enumerate(self.agents):
            agent_state = {
                "id": agent_i,
                "done": self.agent_dones[agent_i],
                "reward": rewards[agent_i] if rewards is not None else None,
                "action": self._json_safe(actions[agent_i]) if actions is not None else None,
                "units": [],
            }
            for unit_i, unit in enumerate(agent.fulfillment_units):
                unit_state = {
                    "id": unit_i,
                    "label": f"A{agent_i}-U{unit_i}",
                    "operation_count": len(unit.operations),
                    "containers": [],
                    "buffers": [],
                    "inventories": [],
                }
                for container_i, container in enumerate(unit.containers):
                    container_state = self._container_state(agent_i, unit_i, container_i, container)
                    unit_state["containers"].append(container_state)
                    if container_state["type"] == "buffer":
                        unit_state["buffers"].append(container_state)
                        total_buffered += container_state["occupied"]
                    elif container_state["type"] == "inventory":
                        unit_state["inventories"].append(container_state)
                        total_inventory += container_state["occupied"]
                    if container_state["capacity"] != -1:
                        total_capacity += container_state["capacity"]
                    orders.extend(container_state.pop("_orders"))
                agent_state["units"].append(unit_state)
            agents.append(agent_state)

        return {
            "step": self.step_count,
            "max_step": self.max_step,
            "timestamp": time.time(),
            "agents": agents,
            "orders": orders[:300],
            "metrics": {
                "agent_count": len(agents),
                "unit_count": sum(len(agent["units"]) for agent in agents),
                "order_count": len(orders),
                "buffered_orders": total_buffered,
                "inventory_orders": total_inventory,
                "finite_capacity": total_capacity,
                "total_reward": sum(rewards) if rewards is not None else None,
            },
        }

    def _container_state(self, agent_i, unit_i, container_i, container):
        resource = container.resource
        capacity = self._container_capacity(container)
        orders = self._orders_from_container(agent_i, unit_i, container_i, container)
        return {
            "id": f"a{agent_i}-u{unit_i}-c{container_i}",
            "type": self._container_type(container),
            "occupied": int(resource.occupied),
            "capacity": int(capacity),
            "resource_constraint": int(resource.constraint),
            "normal_price": float(resource.normal_price),
            "overage_price": float(resource.overage_price),
            "_orders": orders,
        }

    def _container_type(self, container):
        if isinstance(container, Buffer):
            return "buffer"
        if isinstance(container, Inventory):
            return "inventory"
        return container.__class__.__name__.lower()

    def _container_capacity(self, container):
        if isinstance(container, Buffer):
            return container.buffer_len
        if isinstance(container, Inventory):
            return container.inventory_limit
        return container.resource.constraint

    def _orders_from_container(self, agent_i, unit_i, container_i, container):
        if isinstance(container, Buffer):
            orders = []
            for chunk_i, chunk in enumerate(container.buffer):
                for order in chunk:
                    orders.append(self._order_state(agent_i, unit_i, container_i, order, chunk_i))
            return orders
        if isinstance(container, Inventory):
            return [
                self._order_state(agent_i, unit_i, container_i, order, None)
                for order in container.inventory
            ]
        return []

    def _order_state(self, agent_i, unit_i, container_i, order, chunk_i):
        return {
            "id": str(order.reference_number),
            "agent_id": agent_i,
            "unit_id": unit_i,
            "container_id": container_i,
            "chunk": chunk_i,
            "cumulative_price": float(order.cumulative_price),
            "cumulative_time": float(order.cumulative_time),
            "max_price": float(order.max_price),
            "max_time": float(order.max_time),
        }

    def _json_safe(self, value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    def get_agent_obs(self, agent_i):
        return list(self.agents[agent_i].get_obs())

    def get_global_obs(self):
        return [agent.get_obs() for agent in self.agents]

    def get_agent_reward(self, agent_i):
        return self.agents[agent_i].get_reward()

    def get_global_reward(self):
        return [agent.get_reward() for agent in self.agents]

    def _init_action_space(self):
        self.action_space_discrete = MultiAgentActionSpace([
            spaces.Discrete(int(np.prod(multi_discrete_action.nvec)))
            for multi_discrete_action in self.action_space
        ])
        self.md_to_d = []
        self.d_to_md = []
        for agent_i, multi_discrete_action in enumerate(self.action_space):
            possible_action_each_dim = [list(range(int(dim))) for dim in multi_discrete_action.nvec]
            possible_action_tuple = list(itertools.product(*possible_action_each_dim))
            possible_action_multi_discrete = [list(action_tuple) for action_tuple in possible_action_tuple]
            possible_action_discrete = list(range(self.action_space_discrete[agent_i].n))
            md_to_d = {}
            d_to_md = {}
            for md, d in zip(possible_action_multi_discrete, possible_action_discrete):
                md_to_d[str(md)] = str(d)
                d_to_md[str(d)] = str(md)
            self.md_to_d.append(md_to_d)
            self.d_to_md.append(d_to_md)

    def agent_discrete_to_multi_discrete(self, agent_i, discrete):
        return literal_eval(self.d_to_md[agent_i][str(discrete)])

    def agent_multi_discrete_to_discrete(self, agent_i, multi_discrete):
        return int(self.md_to_d[agent_i][str(multi_discrete)])

    def _init_joint_action_space(self):
        joint_action_space_dim = []
        for multi_discrete_action in self.action_space:
            joint_action_space_dim += [int(dim) for dim in multi_discrete_action.nvec]
        joint_action_space_multi_discrete = spaces.MultiDiscrete(joint_action_space_dim)
        self.joint_action_space_discrete = spaces.Discrete(int(np.prod(joint_action_space_multi_discrete.nvec)))

        possible_joint_action_discrete = list(range(self.joint_action_space_discrete.n))
        possible_joint_action_multi_discrete_sep = []
        for multi_discrete_action in self.action_space:
            possible_action_each_dim = [list(range(int(dim))) for dim in multi_discrete_action.nvec]
            possible_action_tuple = list(itertools.product(*possible_action_each_dim))
            possible_action_multi_discrete = [list(action_tuple) for action_tuple in possible_action_tuple]
            possible_joint_action_multi_discrete_sep.append(possible_action_multi_discrete)

        possible_joint_action_multi_discrete_tuple = list(
            itertools.product(*possible_joint_action_multi_discrete_sep)
        )
        possible_joint_action_multi_discrete = [
            list(action_tuple) for action_tuple in possible_joint_action_multi_discrete_tuple
        ]

        self.joint_md_to_d = {}
        self.joint_d_to_md = {}
        for md, d in zip(possible_joint_action_multi_discrete, possible_joint_action_discrete):
            self.joint_md_to_d[str(md)] = str(d)
            self.joint_d_to_md[str(d)] = str(md)

    def joint_discrete_to_multi_discrete(self, discrete):
        return literal_eval(self.joint_d_to_md[str(discrete)])

    def joint_multi_discrete_to_discrete(self, multi_discrete):
        return int(self.joint_md_to_d[str(multi_discrete)])

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
