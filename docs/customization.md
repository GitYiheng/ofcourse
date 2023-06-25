# Environment Customization

Customized fulfillment systems can be constructed in OFCOURSE.
Here, we use [Task 1 (Fulfillment of Physical and Virtual Orders in One System)](../env/define_exp1_env.py) from the paper as an example.

![Fulfillment of Physical and Virtual Orders in One System](../figs/physical_virtual.png)

## Import Modules

```python
import numpy as np
from env.resource import Resource
from env.order import Order
from env.container import Buffer, Inventory
from env.operation import OpStore, OpRoute, OpConsoRoute, OpDispatch
from env.fulfillment_unit import FulfillmentUnit
from env.agent import Agent
from env.order_source import OrderSource
```

## System Variables

Before defining the fulfillment system, we first define the buffer length and inventory capacity.

```python
# ---------- PARAMS ---------- #
buffer_len = 5
inventory_limit = 32
```

## Agents

There are two agents in the fulfillment system. Agent 0 is consisted of 6 fulfillment units and agent 1 is composed of 4 fulfillment units, where they share the first three stages.

```python
# ---------- AGENT 0 ---------- #
agent0 = Agent()
agent0.add_fulfillment_unit(agent0_layer5)
agent0.add_fulfillment_unit(agent0_layer4)
agent0.add_fulfillment_unit(agent0_layer3)
agent0.add_fulfillment_unit(agent0_layer2)
agent0.add_fulfillment_unit(agent0_layer1)
agent0.add_fulfillment_unit(agent0_layer0)

# ---------- AGENT 1 ---------- #
agent1 = Agent()
agent1.add_fulfillment_unit(agent1_layer3)
agent1.add_fulfillment_unit(agent1_layer2)
agent1.add_fulfillment_unit(agent1_layer1)
agent1.add_fulfillment_unit(agent1_layer0)
```

## Fulfillment Stage

Taking the third stage (i.e. the consolidation warehouse) of agent 0 for example, it has two Containers and three Operations.
Each Container has its associated Resource, in which we define Resource before attaching it to the corresponding Container.
Here, one Container is an Inventory and another Container is a Buffer.
In regard to Operations, we have one Operation for storing incoming Orders to the Inventory and two Operations for consolidating and dispatching Orders toward their destinated Buffers.

```python
# 3rd stage in agent 0
agent0_layer3 = FulfillmentUnit()
agent0_layer3_inventory_resource = Resource(constraint=32, normal_price=0.6, overage_price=2.0, occupied=0)
agent0_layer3_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
agent0_layer3_inventory = Inventory(resource=agent0_layer3_inventory_resource, inventory_limit=inventory_limit)
agent0_layer3_buffer0 = Buffer(resource=agent0_layer3_buffer0_resource, buffer_len=buffer_len)
agent0_layer3.add_container(container=agent0_layer3_inventory)
agent0_layer3.add_container(container=agent0_layer3_buffer0)
agent0_layer3_op0 = OpStore(buffers_orig=[agent0_layer3_buffer0], inventory_dest=agent0_layer3_inventory, op_price=0.1, op_time=1)
agent0_layer3_op1 = OpConsoRoute(buffers_orig=[agent0_layer3_buffer0], inventory_orig=agent0_layer3_inventory, buffer_dest=agent0_layer4_buffer0, op_price=4.0, op_time=3)
agent0_layer3_op2 = OpConsoRoute(buffers_orig=[agent0_layer3_buffer0], inventory_orig=agent0_layer3_inventory, buffer_dest=agent0_layer4_buffer1, op_price=8.0, op_time=2)
agent0_layer3.add_operation(operation=agent0_layer3_op0)
agent0_layer3.add_operation(operation=agent0_layer3_op1)
agent0_layer3.add_operation(operation=agent0_layer3_op2)
```
