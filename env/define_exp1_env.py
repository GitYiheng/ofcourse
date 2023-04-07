import numpy as np
from env.resource import Resource
from env.order import Order
from env.container import Buffer, Inventory
from env.operation import OpStore, OpRoute, OpConsoRoute, OpDispatch
from env.fulfillment_unit import FulfillmentUnit
from env.agent import Agent
from env.order_source import OrderSource


def define_exp1_env():
    # ---------- PARAMS ---------- #
    buffer_len = 5
    inventory_limit = 32

    # ---------- AGENT 0 ---------- #
    # 5th layer
    agent0_layer5 = FulfillmentUnit()
    agent0_layer5_target_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer5_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer5_buffer1_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer5_target = Inventory(resource=agent0_layer5_target_resource, inventory_limit=inventory_limit)
    agent0_layer5_buffer0 = Buffer(resource=agent0_layer5_buffer0_resource, buffer_len=buffer_len)
    agent0_layer5_buffer1 = Buffer(resource=agent0_layer5_buffer1_resource, buffer_len=buffer_len)
    agent0_layer5.add_container(container=agent0_layer5_target)
    agent0_layer5.add_container(container=agent0_layer5_buffer0)
    agent0_layer5.add_container(container=agent0_layer5_buffer1)
    agent0_layer5_op0 = OpStore(buffers_orig=[agent0_layer5_buffer0, agent0_layer5_buffer1],
                                inventory_dest=agent0_layer5_target, op_price=0, op_time=0)
    agent0_layer5.add_operation(operation=agent0_layer5_op0)

    # 4th layer
    agent0_layer4 = FulfillmentUnit()
    agent0_layer4_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer4_buffer1_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer4_buffer0 = Buffer(resource=agent0_layer4_buffer0_resource, buffer_len=buffer_len)
    agent0_layer4_buffer1 = Buffer(resource=agent0_layer4_buffer1_resource, buffer_len=buffer_len)
    agent0_layer4.add_container(container=agent0_layer4_buffer0)
    agent0_layer4.add_container(container=agent0_layer4_buffer1)
    agent0_layer4_op0 = OpRoute(buffers_orig=[agent0_layer4_buffer0, agent0_layer4_buffer1],
                                buffer_dest=agent0_layer5_buffer0, op_price=10.0, op_time=2)
    agent0_layer4_op1 = OpRoute(buffers_orig=[agent0_layer4_buffer0, agent0_layer4_buffer1],
                                buffer_dest=agent0_layer5_buffer1, op_price=5.0, op_time=3)
    agent0_layer4.add_operation(operation=agent0_layer4_op0)
    agent0_layer4.add_operation(operation=agent0_layer4_op1)

    # 3rd layer
    agent0_layer3 = FulfillmentUnit()
    agent0_layer3_inventory_resource = Resource(constraint=32, normal_price=0.6, overage_price=2.0, occupied=0)
    agent0_layer3_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer3_inventory = Inventory(resource=agent0_layer3_inventory_resource, inventory_limit=inventory_limit)
    agent0_layer3_buffer0 = Buffer(resource=agent0_layer3_buffer0_resource, buffer_len=buffer_len)
    agent0_layer3.add_container(container=agent0_layer3_inventory)
    agent0_layer3.add_container(container=agent0_layer3_buffer0)
    agent0_layer3_op0 = OpStore(buffers_orig=[agent0_layer3_buffer0], inventory_dest=agent0_layer3_inventory,
                                op_price=0.1, op_time=1)
    agent0_layer3_op1 = OpConsoRoute(buffers_orig=[agent0_layer3_buffer0], inventory_orig=agent0_layer3_inventory,
                                     buffer_dest=agent0_layer4_buffer0, op_price=4.0, op_time=3)
    agent0_layer3_op2 = OpConsoRoute(buffers_orig=[agent0_layer3_buffer0], inventory_orig=agent0_layer3_inventory,
                                     buffer_dest=agent0_layer4_buffer1, op_price=8.0, op_time=2)
    agent0_layer3.add_operation(operation=agent0_layer3_op0)
    agent0_layer3.add_operation(operation=agent0_layer3_op1)
    agent0_layer3.add_operation(operation=agent0_layer3_op2)

    # 2nd layer
    agent0_layer2 = FulfillmentUnit()
    agent0_layer2_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer2_buffer0 = Buffer(resource=agent0_layer2_buffer0_resource, buffer_len=buffer_len)
    agent0_layer2.add_container(container=agent0_layer2_buffer0)
    agent0_layer2_op0 = OpRoute(buffers_orig=[agent0_layer2_buffer0], buffer_dest=agent0_layer3_buffer0, op_price=5.0,
                                op_time=3)
    agent0_layer2.add_operation(operation=agent0_layer2_op0)

    # 1st layer
    agent0_layer1 = FulfillmentUnit()
    agent0_layer1_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer1_buffer0 = Buffer(resource=agent0_layer1_buffer0_resource, buffer_len=buffer_len)
    agent0_layer1.add_container(container=agent0_layer1_buffer0)
    agent0_layer1_op0 = OpRoute(buffers_orig=[agent0_layer1_buffer0], buffer_dest=agent0_layer2_buffer0, op_price=2.0,
                                op_time=1)
    agent0_layer1.add_operation(operation=agent0_layer1_op0)

    # 0th layer
    agent0_layer0 = FulfillmentUnit()
    agent0_layer0_order_queue = np.tile(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0]).astype(bool), 15)
    agent0_layer0_order_source = OrderSource(order_queue=agent0_layer0_order_queue)
    agent0_layer0_op0 = OpDispatch(order_src=agent0_layer0_order_source, buffer_dest=agent0_layer1_buffer0, op_price=0,
                                   op_time=1)
    agent0_layer0.add_operation(operation=agent0_layer0_op0)

    # agent 0
    agent0 = Agent()
    agent0.add_fulfillment_unit(agent0_layer5)
    agent0.add_fulfillment_unit(agent0_layer4)
    agent0.add_fulfillment_unit(agent0_layer3)
    agent0.add_fulfillment_unit(agent0_layer2)
    agent0.add_fulfillment_unit(agent0_layer1)
    agent0.add_fulfillment_unit(agent0_layer0)

    # ---------- AGENT 1 ---------- #
    # 3rd layer
    agent1_layer3 = FulfillmentUnit()
    agent1_layer3_target_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent1_layer3_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent1_layer3_target = Inventory(resource=agent1_layer3_target_resource, inventory_limit=inventory_limit)
    agent1_layer3_buffer0 = Buffer(resource=agent1_layer3_buffer0_resource, buffer_len=buffer_len)
    agent1_layer3.add_container(container=agent1_layer3_target)
    agent1_layer3.add_container(container=agent1_layer3_buffer0)
    agent1_layer3_op0 = OpStore(buffers_orig=[agent1_layer3_buffer0], inventory_dest=agent1_layer3_target, op_price=0,
                                op_time=0)
    agent1_layer3.add_operation(operation=agent1_layer3_op0)

    # 2nd layer
    agent1_layer2 = FulfillmentUnit()
    agent1_layer2_buffer0 = Buffer(resource=agent0_layer2_buffer0_resource, buffer_len=buffer_len)
    agent1_layer2.add_container(container=agent1_layer2_buffer0)
    agent1_layer2_op0 = OpRoute(buffers_orig=[agent1_layer2_buffer0], buffer_dest=agent1_layer3_buffer0, op_price=2.0,
                                op_time=1)
    agent1_layer2.add_operation(operation=agent1_layer2_op0)

    # 1st layer
    agent1_layer1 = FulfillmentUnit()
    agent1_layer1_buffer0 = Buffer(resource=agent0_layer1_buffer0_resource, buffer_len=buffer_len)
    agent1_layer1.add_container(container=agent1_layer1_buffer0)
    agent1_layer1_op0 = OpRoute(buffers_orig=[agent1_layer1_buffer0], buffer_dest=agent1_layer2_buffer0, op_price=2.0,
                                op_time=1)
    agent1_layer1.add_operation(operation=agent1_layer1_op0)

    # 0th layer
    agent1_layer0 = FulfillmentUnit()
    agent1_layer0_order_queue = np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(bool), 15)
    agent1_layer0_order_source = OrderSource(order_queue=agent1_layer0_order_queue)
    agent1_layer0_op0 = OpDispatch(order_src=agent1_layer0_order_source, buffer_dest=agent1_layer1_buffer0, op_price=0,
                                   op_time=1)
    agent1_layer0.add_operation(operation=agent1_layer0_op0)

    # agent 1
    agent1 = Agent()
    agent1.add_fulfillment_unit(agent1_layer3)
    agent1.add_fulfillment_unit(agent1_layer2)
    agent1.add_fulfillment_unit(agent1_layer1)
    agent1.add_fulfillment_unit(agent1_layer0)

    return [agent0, agent1]
