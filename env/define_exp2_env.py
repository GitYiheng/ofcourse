import numpy as np
from env.resource import Resource
from env.order import Order
from env.container import Buffer, Inventory
from env.operation import OpStore, OpRoute, OpConsoRoute, OpDispatch
from env.fulfillment_unit import FulfillmentUnit
from env.agent import Agent
from env.order_source import OrderSource


def define_exp2_env():
    # ---------- PARAMS ---------- #
    buffer_len = 5
    inventory_limit = 32

    # ---------- AGENT 0 ---------- #
    # seller0-to-customer0 route0

    # agent0 - 12th layer [order delivered]
    agent0_layer12 = FulfillmentUnit()
    agent0_layer12_target_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer12_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer12_buffer1_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer12_target = Inventory(resource=agent0_layer12_target_resource, inventory_limit=inventory_limit)
    agent0_layer12_buffer0 = Buffer(resource=agent0_layer12_buffer0_resource, buffer_len=buffer_len)
    agent0_layer12_buffer1 = Buffer(resource=agent0_layer12_buffer1_resource, buffer_len=buffer_len)
    agent0_layer12.add_container(container=agent0_layer12_target)
    agent0_layer12.add_container(container=agent0_layer12_buffer0)
    agent0_layer12.add_container(container=agent0_layer12_buffer1)
    agent0_layer12_op0 = OpStore(buffers_orig=[agent0_layer12_buffer0, agent0_layer12_buffer1],
                                 inventory_dest=agent0_layer12_target, op_price=0, op_time=0)
    agent0_layer12.add_operation(operation=agent0_layer12_op0)

    # agent0 - 11th layer [last-mile warehouse]
    agent0_layer11 = FulfillmentUnit()
    agent0_layer11_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer11_buffer0 = Buffer(resource=agent0_layer11_buffer0_resource, buffer_len=buffer_len)
    agent0_layer11.add_container(container=agent0_layer11_buffer0)
    agent0_layer11_op0 = OpRoute(buffers_orig=[agent0_layer11_buffer0], buffer_dest=agent0_layer12_buffer0,
                                 op_price=10.0, op_time=1)
    agent0_layer11_op1 = OpRoute(buffers_orig=[agent0_layer11_buffer0], buffer_dest=agent0_layer12_buffer1,
                                 op_price=5.0, op_time=2)
    agent0_layer11.add_operation(operation=agent0_layer11_op0)
    agent0_layer11.add_operation(operation=agent0_layer11_op1)

    # agent0 - 10th layer [regional hub]
    agent0_layer10 = FulfillmentUnit()
    agent0_layer10_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer10_buffer0 = Buffer(resource=agent0_layer10_buffer0_resource, buffer_len=buffer_len)
    agent0_layer10.add_container(container=agent0_layer10_buffer0)
    agent0_layer10_op0 = OpRoute(buffers_orig=[agent0_layer10_buffer0], buffer_dest=agent0_layer11_buffer0,
                                 op_price=5.0, op_time=2)
    agent0_layer10.add_operation(operation=agent0_layer10_op0)

    # agent0 - 9th layer [freight station arrival]
    agent0_layer9 = FulfillmentUnit()
    agent0_layer9_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer9_buffer0 = Buffer(resource=agent0_layer9_buffer0_resource, buffer_len=buffer_len)
    agent0_layer9.add_container(container=agent0_layer9_buffer0)
    agent0_layer9_op0 = OpRoute(buffers_orig=[agent0_layer9_buffer0], buffer_dest=agent0_layer10_buffer0, op_price=5.0,
                                op_time=2)
    agent0_layer9.add_operation(operation=agent0_layer9_op0)

    # agent0 - 8th layer [freight station departure]
    agent0_layer8 = FulfillmentUnit()
    agent0_layer8_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer8_buffer0 = Buffer(resource=agent0_layer8_buffer0_resource, buffer_len=buffer_len)
    agent0_layer8.add_container(container=agent0_layer8_buffer0)
    agent0_layer8_op0 = OpRoute(buffers_orig=[agent0_layer8_buffer0], buffer_dest=agent0_layer9_buffer0, op_price=50.0,
                                op_time=4)
    agent0_layer8.add_operation(operation=agent0_layer8_op0)

    # agent0 - 7th layer [linehaul warehouse]
    agent0_layer7 = FulfillmentUnit()
    agent0_layer7_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer7_buffer0 = Buffer(resource=agent0_layer7_buffer0_resource, buffer_len=buffer_len)
    agent0_layer7.add_container(container=agent0_layer7_buffer0)
    agent0_layer7_op0 = OpRoute(buffers_orig=[agent0_layer7_buffer0], buffer_dest=agent0_layer8_buffer0, op_price=5.0,
                                op_time=2)
    agent0_layer7.add_operation(operation=agent0_layer7_op0)

    # agent0 - 6th layer [distribution center of consolidation warehouse]
    agent0_layer6 = FulfillmentUnit()
    agent0_layer6_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer6_buffer0 = Buffer(resource=agent0_layer6_buffer0_resource, buffer_len=buffer_len)
    agent0_layer6.add_container(container=agent0_layer6_buffer0)
    agent0_layer6_op0 = OpRoute(buffers_orig=[agent0_layer6_buffer0], buffer_dest=agent0_layer7_buffer0, op_price=5.0,
                                op_time=2)
    agent0_layer6.add_operation(operation=agent0_layer6_op0)

    # agent0 - 5th layer [consolidation warehouse]
    agent0_layer5 = FulfillmentUnit()
    agent0_layer5_inventory_resource = Resource(constraint=92, normal_price=1.0, overage_price=2.0, occupied=0)
    agent0_layer5_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer5_inventory = Inventory(resource=agent0_layer5_inventory_resource, inventory_limit=inventory_limit)
    agent0_layer5_buffer0 = Buffer(resource=agent0_layer5_buffer0_resource, buffer_len=buffer_len)
    agent0_layer5.add_container(container=agent0_layer5_inventory)
    agent0_layer5.add_container(container=agent0_layer5_buffer0)
    agent0_layer5_op0 = OpStore(buffers_orig=[agent0_layer5_buffer0], inventory_dest=agent0_layer5_inventory,
                                op_price=0.1, op_time=1)
    agent0_layer5_op1 = OpConsoRoute(buffers_orig=[agent0_layer5_buffer0], inventory_orig=agent0_layer5_inventory,
                                     buffer_dest=agent0_layer6_buffer0, op_price=5.0, op_time=2)
    agent0_layer5.add_operation(operation=agent0_layer5_op0)
    agent0_layer5.add_operation(operation=agent0_layer5_op1)

    # agent0 - 4th layer [distribution center]
    agent0_layer4 = FulfillmentUnit()
    agent0_layer4_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer4_buffer0 = Buffer(resource=agent0_layer4_buffer0_resource, buffer_len=buffer_len)
    agent0_layer4.add_container(container=agent0_layer4_buffer0)
    agent0_layer4_op0 = OpRoute(buffers_orig=[agent0_layer4_buffer0], buffer_dest=agent0_layer5_buffer0, op_price=5.0,
                                op_time=2)
    agent0_layer4.add_operation(operation=agent0_layer4_op0)

    # agent0 - 3rd layer [collection warehouse]
    agent0_layer3 = FulfillmentUnit()
    agent0_layer3_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer3_buffer1_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer3_buffer0 = Buffer(resource=agent0_layer3_buffer0_resource, buffer_len=buffer_len)
    agent0_layer3_buffer1 = Buffer(resource=agent0_layer3_buffer1_resource, buffer_len=buffer_len)
    agent0_layer3.add_container(container=agent0_layer3_buffer0)
    agent0_layer3.add_container(container=agent0_layer3_buffer1)
    agent0_layer3_op0 = OpRoute(buffers_orig=[agent0_layer3_buffer0, agent0_layer3_buffer1],
                                buffer_dest=agent0_layer4_buffer0, op_price=5.0, op_time=2)
    agent0_layer3.add_operation(operation=agent0_layer3_op0)

    # agent0 - 2nd layer [seller]
    agent0_layer2 = FulfillmentUnit()
    agent0_layer2_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer2_buffer0 = Buffer(resource=agent0_layer2_buffer0_resource, buffer_len=buffer_len)
    agent0_layer2.add_container(container=agent0_layer2_buffer0)
    agent0_layer2_op0 = OpRoute(buffers_orig=[agent0_layer2_buffer0], buffer_dest=agent0_layer3_buffer0, op_price=10.0,
                                op_time=1)
    agent0_layer2_op1 = OpRoute(buffers_orig=[agent0_layer2_buffer0], buffer_dest=agent0_layer3_buffer1, op_price=5.0,
                                op_time=2)
    agent0_layer2.add_operation(operation=agent0_layer2_op0)
    agent0_layer2.add_operation(operation=agent0_layer2_op1)

    # agent0 - 1st layer [e-commerce platform]
    agent0_layer1 = FulfillmentUnit()
    agent0_layer1_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent0_layer1_buffer0 = Buffer(resource=agent0_layer1_buffer0_resource, buffer_len=buffer_len)
    agent0_layer1.add_container(container=agent0_layer1_buffer0)
    agent0_layer1_op0 = OpRoute(buffers_orig=[agent0_layer1_buffer0], buffer_dest=agent0_layer2_buffer0, op_price=5.0,
                                op_time=1)
    agent0_layer1.add_operation(operation=agent0_layer1_op0)

    # agent0 - 0th layer [order placement]
    agent0_layer0 = FulfillmentUnit()
    agent0_layer0_order_queue = np.tile(np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0]).astype(bool), 15)
    agent0_layer0_order_source = OrderSource(order_queue=agent0_layer0_order_queue)
    agent0_layer0_op0 = OpDispatch(order_src=agent0_layer0_order_source, buffer_dest=agent0_layer1_buffer0, op_price=0,
                                   op_time=1)
    agent0_layer0.add_operation(operation=agent0_layer0_op0)

    # agent0 assembling
    agent0 = Agent()
    agent0.add_fulfillment_unit(agent0_layer12)
    agent0.add_fulfillment_unit(agent0_layer11)
    agent0.add_fulfillment_unit(agent0_layer10)
    agent0.add_fulfillment_unit(agent0_layer9)
    agent0.add_fulfillment_unit(agent0_layer8)
    agent0.add_fulfillment_unit(agent0_layer7)
    agent0.add_fulfillment_unit(agent0_layer6)
    agent0.add_fulfillment_unit(agent0_layer5)
    agent0.add_fulfillment_unit(agent0_layer4)
    agent0.add_fulfillment_unit(agent0_layer3)
    agent0.add_fulfillment_unit(agent0_layer2)
    agent0.add_fulfillment_unit(agent0_layer1)
    agent0.add_fulfillment_unit(agent0_layer0)

    # ---------- AGENT 1 ---------- #
    # seller0-to-customer0 route1

    # agent1 - 12th layer [order delivered]
    agent1_layer12 = FulfillmentUnit()
    agent1_layer12_target = Inventory(resource=agent0_layer12_target_resource, inventory_limit=inventory_limit)
    agent1_layer12_buffer0 = Buffer(resource=agent0_layer12_buffer0_resource, buffer_len=buffer_len)
    agent1_layer12_buffer1 = Buffer(resource=agent0_layer12_buffer1_resource, buffer_len=buffer_len)
    agent1_layer12.add_container(container=agent1_layer12_target)
    agent1_layer12.add_container(container=agent1_layer12_buffer0)
    agent1_layer12.add_container(container=agent1_layer12_buffer1)
    agent1_layer12_op0 = OpStore(buffers_orig=[agent1_layer12_buffer0, agent1_layer12_buffer1],
                                 inventory_dest=agent1_layer12_target, op_price=0, op_time=0)
    agent1_layer12.add_operation(operation=agent1_layer12_op0)

    # agent1 - 11th layer [last-mile warehouse]
    agent1_layer11 = FulfillmentUnit()
    agent1_layer11_buffer0 = Buffer(resource=agent0_layer11_buffer0_resource, buffer_len=buffer_len)
    agent1_layer11.add_container(container=agent1_layer11_buffer0)
    agent1_layer11_op0 = OpRoute(buffers_orig=[agent1_layer11_buffer0], buffer_dest=agent1_layer12_buffer0,
                                 op_price=10.0, op_time=1)
    agent1_layer11_op1 = OpRoute(buffers_orig=[agent1_layer11_buffer0], buffer_dest=agent1_layer12_buffer1,
                                 op_price=5.0, op_time=2)
    agent1_layer11.add_operation(operation=agent1_layer11_op0)
    agent1_layer11.add_operation(operation=agent1_layer11_op1)

    # agent1 - 10th layer [regional hub]
    agent1_layer10 = FulfillmentUnit()
    agent1_layer10_buffer0 = Buffer(resource=agent0_layer10_buffer0_resource, buffer_len=buffer_len)
    agent1_layer10.add_container(container=agent1_layer10_buffer0)
    agent1_layer10_op0 = OpRoute(buffers_orig=[agent1_layer10_buffer0], buffer_dest=agent1_layer11_buffer0,
                                 op_price=5.0, op_time=2)
    agent1_layer10.add_operation(operation=agent1_layer10_op0)

    # agent1 - 9th layer [freight station arrival]
    agent1_layer9 = FulfillmentUnit()
    agent1_layer9_buffer0 = Buffer(resource=agent0_layer9_buffer0_resource, buffer_len=buffer_len)
    agent1_layer9.add_container(container=agent1_layer9_buffer0)
    agent1_layer9_op0 = OpRoute(buffers_orig=[agent1_layer9_buffer0], buffer_dest=agent1_layer10_buffer0, op_price=5.0,
                                op_time=2)
    agent1_layer9.add_operation(operation=agent1_layer9_op0)

    # agent1 - 8th layer [freight station departure]
    agent1_layer8 = FulfillmentUnit()
    agent1_layer8_buffer0 = Buffer(resource=agent0_layer8_buffer0_resource, buffer_len=buffer_len)
    agent1_layer8.add_container(container=agent1_layer8_buffer0)
    agent1_layer8_op0 = OpRoute(buffers_orig=[agent1_layer8_buffer0], buffer_dest=agent1_layer9_buffer0, op_price=50.0,
                                op_time=4)
    agent1_layer8.add_operation(operation=agent1_layer8_op0)

    # agent1 - 7th layer [linehaul warehouse]
    agent1_layer7 = FulfillmentUnit()
    agent1_layer7_buffer0 = Buffer(resource=agent0_layer7_buffer0_resource, buffer_len=buffer_len)
    agent1_layer7.add_container(container=agent1_layer7_buffer0)
    agent1_layer7_op0 = OpRoute(buffers_orig=[agent1_layer7_buffer0], buffer_dest=agent1_layer8_buffer0, op_price=5.0,
                                op_time=2)
    agent1_layer7.add_operation(operation=agent1_layer7_op0)

    # agent1 - 4th layer [distribution center]
    agent1_layer4 = FulfillmentUnit()
    agent1_layer4_buffer0 = Buffer(resource=agent0_layer4_buffer0_resource, buffer_len=buffer_len)
    agent1_layer4.add_container(container=agent1_layer4_buffer0)
    agent1_layer4_op0 = OpRoute(buffers_orig=[agent1_layer4_buffer0], buffer_dest=agent1_layer7_buffer0, op_price=5.0,
                                op_time=2)
    agent1_layer4.add_operation(operation=agent1_layer4_op0)

    # agent1 - 3rd layer [collection warehouse]
    agent1_layer3 = FulfillmentUnit()
    agent1_layer3_buffer0 = Buffer(resource=agent0_layer3_buffer0_resource, buffer_len=buffer_len)
    agent1_layer3_buffer1 = Buffer(resource=agent0_layer3_buffer1_resource, buffer_len=buffer_len)
    agent1_layer3.add_container(container=agent1_layer3_buffer0)
    agent1_layer3.add_container(container=agent1_layer3_buffer1)
    agent1_layer3_op0 = OpRoute(buffers_orig=[agent1_layer3_buffer0, agent1_layer3_buffer1],
                                buffer_dest=agent1_layer4_buffer0, op_price=5.0, op_time=2)
    agent1_layer3.add_operation(operation=agent1_layer3_op0)

    # agent1 - 2nd layer [seller]
    agent1_layer2 = FulfillmentUnit()
    agent1_layer2_buffer0 = Buffer(resource=agent0_layer2_buffer0_resource, buffer_len=buffer_len)
    agent1_layer2.add_container(container=agent1_layer2_buffer0)
    agent1_layer2_op0 = OpRoute(buffers_orig=[agent1_layer2_buffer0], buffer_dest=agent1_layer3_buffer0, op_price=10.0,
                                op_time=1)
    agent1_layer2_op1 = OpRoute(buffers_orig=[agent1_layer2_buffer0], buffer_dest=agent1_layer3_buffer1, op_price=5.0,
                                op_time=2)
    agent1_layer2.add_operation(operation=agent1_layer2_op0)
    agent1_layer2.add_operation(operation=agent1_layer2_op1)

    # agent1 - 1st layer [e-commerce platform]
    agent1_layer1 = FulfillmentUnit()
    agent1_layer1_buffer0 = Buffer(resource=agent0_layer1_buffer0_resource, buffer_len=buffer_len)
    agent1_layer1.add_container(container=agent1_layer1_buffer0)
    agent1_layer1_op0 = OpRoute(buffers_orig=[agent1_layer1_buffer0], buffer_dest=agent1_layer2_buffer0, op_price=5.0,
                                op_time=1)
    agent1_layer1.add_operation(operation=agent1_layer1_op0)

    # agent0 - 0th layer [order placement]
    agent1_layer0 = FulfillmentUnit()
    agent1_layer0_order_queue = np.tile(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).astype(bool), 15)
    agent1_layer0_order_source = OrderSource(order_queue=agent1_layer0_order_queue)
    agent1_layer0_op0 = OpDispatch(order_src=agent1_layer0_order_source, buffer_dest=agent1_layer1_buffer0, op_price=0,
                                   op_time=1)
    agent1_layer0.add_operation(operation=agent1_layer0_op0)

    # agent1 assembling
    agent1 = Agent()
    agent1.add_fulfillment_unit(agent1_layer12)
    agent1.add_fulfillment_unit(agent1_layer11)
    agent1.add_fulfillment_unit(agent1_layer10)
    agent1.add_fulfillment_unit(agent1_layer9)
    agent1.add_fulfillment_unit(agent1_layer8)
    agent1.add_fulfillment_unit(agent1_layer7)
    agent1.add_fulfillment_unit(agent1_layer4)
    agent1.add_fulfillment_unit(agent1_layer3)
    agent1.add_fulfillment_unit(agent1_layer2)
    agent1.add_fulfillment_unit(agent1_layer1)
    agent1.add_fulfillment_unit(agent1_layer0)

    # ---------- AGENT 2 ---------- #
    # seller0-to-customer1 route0

    # agent2 - 12th layer [order delivered]
    agent2_layer12 = FulfillmentUnit()
    agent2_layer12_target_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent2_layer12_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent2_layer12_buffer1_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent2_layer12_target = Inventory(resource=agent2_layer12_target_resource, inventory_limit=inventory_limit)
    agent2_layer12_buffer0 = Buffer(resource=agent2_layer12_buffer0_resource, buffer_len=buffer_len)
    agent2_layer12_buffer1 = Buffer(resource=agent2_layer12_buffer1_resource, buffer_len=buffer_len)
    agent2_layer12.add_container(container=agent2_layer12_target)
    agent2_layer12.add_container(container=agent2_layer12_buffer0)
    agent2_layer12.add_container(container=agent2_layer12_buffer1)
    agent2_layer12_op0 = OpStore(buffers_orig=[agent2_layer12_buffer0, agent2_layer12_buffer1],
                                 inventory_dest=agent2_layer12_target, op_price=0, op_time=0)
    agent2_layer12.add_operation(operation=agent2_layer12_op0)

    # agent2 - 11th layer [last-mile warehouse]
    agent2_layer11 = FulfillmentUnit()
    agent2_layer11_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent2_layer11_buffer0 = Buffer(resource=agent2_layer11_buffer0_resource, buffer_len=buffer_len)
    agent2_layer11.add_container(container=agent2_layer11_buffer0)
    agent2_layer11_op0 = OpRoute(buffers_orig=[agent2_layer11_buffer0], buffer_dest=agent2_layer12_buffer0,
                                 op_price=12.0, op_time=1)
    agent2_layer11_op1 = OpRoute(buffers_orig=[agent2_layer11_buffer0], buffer_dest=agent2_layer12_buffer1,
                                 op_price=6.0, op_time=2)
    agent2_layer11.add_operation(operation=agent2_layer11_op0)
    agent2_layer11.add_operation(operation=agent2_layer11_op1)

    # agent2 - 10th layer [regional hub]
    agent2_layer10 = FulfillmentUnit()
    agent2_layer10_buffer0 = Buffer(resource=agent0_layer10_buffer0_resource, buffer_len=buffer_len)
    agent2_layer10.add_container(container=agent2_layer10_buffer0)
    agent2_layer10_op0 = OpRoute(buffers_orig=[agent2_layer10_buffer0], buffer_dest=agent2_layer11_buffer0,
                                 op_price=5.0, op_time=2)
    agent2_layer10.add_operation(operation=agent2_layer10_op0)

    # agent2 - 9th layer [freight station arrival]
    agent2_layer9 = FulfillmentUnit()
    agent2_layer9_buffer0 = Buffer(resource=agent0_layer9_buffer0_resource, buffer_len=buffer_len)
    agent2_layer9.add_container(container=agent2_layer9_buffer0)
    agent2_layer9_op0 = OpRoute(buffers_orig=[agent2_layer9_buffer0], buffer_dest=agent2_layer10_buffer0, op_price=5.0,
                                op_time=2)
    agent2_layer9.add_operation(operation=agent2_layer9_op0)

    # agent2 - 8th layer [freight station departure]
    agent2_layer8 = FulfillmentUnit()
    agent2_layer8_buffer0 = Buffer(resource=agent0_layer8_buffer0_resource, buffer_len=buffer_len)
    agent2_layer8.add_container(container=agent2_layer8_buffer0)
    agent2_layer8_op0 = OpRoute(buffers_orig=[agent2_layer8_buffer0], buffer_dest=agent2_layer9_buffer0, op_price=50.0,
                                op_time=4)
    agent2_layer8.add_operation(operation=agent2_layer8_op0)

    # agent2 - 7th layer [linehaul warehouse]
    agent2_layer7 = FulfillmentUnit()
    agent2_layer7_buffer0 = Buffer(resource=agent0_layer7_buffer0_resource, buffer_len=buffer_len)
    agent2_layer7.add_container(container=agent2_layer7_buffer0)
    agent2_layer7_op0 = OpRoute(buffers_orig=[agent2_layer7_buffer0], buffer_dest=agent2_layer8_buffer0, op_price=5.0,
                                op_time=2)
    agent2_layer7.add_operation(operation=agent2_layer7_op0)

    # agent2 - 6th layer [distribution center of consolidation warehouse]
    agent2_layer6 = FulfillmentUnit()
    agent2_layer6_buffer0 = Buffer(resource=agent0_layer6_buffer0_resource, buffer_len=buffer_len)
    agent2_layer6.add_container(container=agent2_layer6_buffer0)
    agent2_layer6_op0 = OpRoute(buffers_orig=[agent2_layer6_buffer0], buffer_dest=agent2_layer7_buffer0, op_price=5.0,
                                op_time=2)
    agent2_layer6.add_operation(operation=agent2_layer6_op0)

    # agent2 - 5th layer [consolidation warehouse]
    agent2_layer5 = FulfillmentUnit()
    agent2_layer5_inventory = Inventory(resource=agent0_layer5_inventory_resource, inventory_limit=inventory_limit)
    agent2_layer5_buffer0 = Buffer(resource=agent0_layer5_buffer0_resource, buffer_len=buffer_len)
    agent2_layer5.add_container(container=agent2_layer5_inventory)
    agent2_layer5.add_container(container=agent2_layer5_buffer0)
    agent2_layer5_op0 = OpStore(buffers_orig=[agent2_layer5_buffer0], inventory_dest=agent2_layer5_inventory,
                                op_price=0.1, op_time=1)
    agent2_layer5_op1 = OpConsoRoute(buffers_orig=[agent2_layer5_buffer0], inventory_orig=agent2_layer5_inventory,
                                     buffer_dest=agent2_layer6_buffer0, op_price=5.0, op_time=2)
    agent2_layer5.add_operation(operation=agent2_layer5_op0)
    agent2_layer5.add_operation(operation=agent2_layer5_op1)

    # agent2 - 4th layer [distribution center]
    agent2_layer4 = FulfillmentUnit()
    agent2_layer4_buffer0 = Buffer(resource=agent0_layer4_buffer0_resource, buffer_len=buffer_len)
    agent2_layer4.add_container(container=agent2_layer4_buffer0)
    agent2_layer4_op0 = OpRoute(buffers_orig=[agent2_layer4_buffer0], buffer_dest=agent2_layer5_buffer0, op_price=5.0,
                                op_time=2)
    agent2_layer4.add_operation(operation=agent2_layer4_op0)

    # agent2 - 3rd layer [collection warehouse]
    agent2_layer3 = FulfillmentUnit()
    agent2_layer3_buffer0 = Buffer(resource=agent0_layer3_buffer0_resource, buffer_len=buffer_len)
    agent2_layer3_buffer1 = Buffer(resource=agent0_layer3_buffer1_resource, buffer_len=buffer_len)
    agent2_layer3.add_container(container=agent2_layer3_buffer0)
    agent2_layer3.add_container(container=agent2_layer3_buffer1)
    agent2_layer3_op0 = OpRoute(buffers_orig=[agent2_layer3_buffer0, agent2_layer3_buffer1],
                                buffer_dest=agent2_layer4_buffer0, op_price=5.0, op_time=2)
    agent2_layer3.add_operation(operation=agent2_layer3_op0)

    # agent2 - 2nd layer [seller]
    agent2_layer2 = FulfillmentUnit()
    agent2_layer2_buffer0 = Buffer(resource=agent0_layer2_buffer0_resource, buffer_len=buffer_len)
    agent2_layer2.add_container(container=agent2_layer2_buffer0)
    agent2_layer2_op0 = OpRoute(buffers_orig=[agent2_layer2_buffer0], buffer_dest=agent2_layer3_buffer0, op_price=10.0,
                                op_time=1)
    agent2_layer2_op1 = OpRoute(buffers_orig=[agent2_layer2_buffer0], buffer_dest=agent2_layer3_buffer1, op_price=5.0,
                                op_time=2)
    agent2_layer2.add_operation(operation=agent2_layer2_op0)
    agent2_layer2.add_operation(operation=agent2_layer2_op1)

    # agent2 - 1st layer [e-commerce platform]
    agent2_layer1 = FulfillmentUnit()
    agent2_layer1_buffer0 = Buffer(resource=agent0_layer1_buffer0_resource, buffer_len=buffer_len)
    agent2_layer1.add_container(container=agent2_layer1_buffer0)
    agent2_layer1_op0 = OpRoute(buffers_orig=[agent2_layer1_buffer0], buffer_dest=agent2_layer2_buffer0, op_price=5.0,
                                op_time=1)
    agent2_layer1.add_operation(operation=agent2_layer1_op0)

    # agent2 - 0th layer [order placement]
    agent2_layer0 = FulfillmentUnit()
    agent2_layer0_order_queue = np.tile(np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0]).astype(bool), 15)
    agent2_layer0_order_source = OrderSource(order_queue=agent2_layer0_order_queue)
    agent2_layer0_op0 = OpDispatch(order_src=agent2_layer0_order_source, buffer_dest=agent2_layer1_buffer0, op_price=0,
                                   op_time=1)
    agent2_layer0.add_operation(operation=agent2_layer0_op0)

    # agent2 assembling
    agent2 = Agent()
    agent2.add_fulfillment_unit(agent2_layer12)
    agent2.add_fulfillment_unit(agent2_layer11)
    agent2.add_fulfillment_unit(agent2_layer10)
    agent2.add_fulfillment_unit(agent2_layer9)
    agent2.add_fulfillment_unit(agent2_layer8)
    agent2.add_fulfillment_unit(agent2_layer7)
    agent2.add_fulfillment_unit(agent2_layer6)
    agent2.add_fulfillment_unit(agent2_layer5)
    agent2.add_fulfillment_unit(agent2_layer4)
    agent2.add_fulfillment_unit(agent2_layer3)
    agent2.add_fulfillment_unit(agent2_layer2)
    agent2.add_fulfillment_unit(agent2_layer1)
    agent2.add_fulfillment_unit(agent2_layer0)

    # ---------- AGENT 3 ---------- #
    # seller0-to-customer1 route1

    # agent3 - 12th layer [order delivered]
    agent3_layer12 = FulfillmentUnit()
    agent3_layer12_target = Inventory(resource=agent2_layer12_target_resource, inventory_limit=inventory_limit)
    agent3_layer12_buffer0 = Buffer(resource=agent2_layer12_buffer0_resource, buffer_len=buffer_len)
    agent3_layer12_buffer1 = Buffer(resource=agent2_layer12_buffer1_resource, buffer_len=buffer_len)
    agent3_layer12.add_container(container=agent3_layer12_target)
    agent3_layer12.add_container(container=agent3_layer12_buffer0)
    agent3_layer12.add_container(container=agent3_layer12_buffer1)
    agent3_layer12_op0 = OpStore(buffers_orig=[agent3_layer12_buffer0, agent3_layer12_buffer1],
                                 inventory_dest=agent3_layer12_target, op_price=0, op_time=0)
    agent3_layer12.add_operation(operation=agent3_layer12_op0)

    # agent3 - 11th layer [last-mile warehouse]
    agent3_layer11 = FulfillmentUnit()
    agent3_layer11_buffer0 = Buffer(resource=agent2_layer11_buffer0_resource, buffer_len=buffer_len)
    agent3_layer11.add_container(container=agent3_layer11_buffer0)
    agent3_layer11_op0 = OpRoute(buffers_orig=[agent3_layer11_buffer0], buffer_dest=agent3_layer12_buffer0,
                                 op_price=12.0, op_time=1)
    agent3_layer11_op1 = OpRoute(buffers_orig=[agent3_layer11_buffer0], buffer_dest=agent3_layer12_buffer1,
                                 op_price=6.0, op_time=2)
    agent3_layer11.add_operation(operation=agent3_layer11_op0)
    agent3_layer11.add_operation(operation=agent3_layer11_op1)

    # agent3 - 10th layer [regional hub]
    agent3_layer10 = FulfillmentUnit()
    agent3_layer10_buffer0 = Buffer(resource=agent0_layer10_buffer0_resource, buffer_len=buffer_len)
    agent3_layer10.add_container(container=agent3_layer10_buffer0)
    agent3_layer10_op0 = OpRoute(buffers_orig=[agent3_layer10_buffer0], buffer_dest=agent3_layer11_buffer0,
                                 op_price=5.0, op_time=2)
    agent3_layer10.add_operation(operation=agent3_layer10_op0)

    # agent3 - 9th layer [freight station arrival]
    agent3_layer9 = FulfillmentUnit()
    agent3_layer9_buffer0 = Buffer(resource=agent0_layer9_buffer0_resource, buffer_len=buffer_len)
    agent3_layer9.add_container(container=agent3_layer9_buffer0)
    agent3_layer9_op0 = OpRoute(buffers_orig=[agent3_layer9_buffer0], buffer_dest=agent3_layer10_buffer0, op_price=5.0,
                                op_time=2)
    agent3_layer9.add_operation(operation=agent3_layer9_op0)

    # agent3 - 8th layer [freight station departure]
    agent3_layer8 = FulfillmentUnit()
    agent3_layer8_buffer0 = Buffer(resource=agent0_layer8_buffer0_resource, buffer_len=buffer_len)
    agent3_layer8.add_container(container=agent3_layer8_buffer0)
    agent3_layer8_op0 = OpRoute(buffers_orig=[agent3_layer8_buffer0], buffer_dest=agent3_layer9_buffer0, op_price=50.0,
                                op_time=4)
    agent3_layer8.add_operation(operation=agent3_layer8_op0)

    # agent3 - 7th layer [linehaul warehouse]
    agent3_layer7 = FulfillmentUnit()
    agent3_layer7_buffer0 = Buffer(resource=agent0_layer7_buffer0_resource, buffer_len=buffer_len)
    agent3_layer7.add_container(container=agent3_layer7_buffer0)
    agent3_layer7_op0 = OpRoute(buffers_orig=[agent3_layer7_buffer0], buffer_dest=agent3_layer8_buffer0, op_price=5.0,
                                op_time=2)
    agent3_layer7.add_operation(operation=agent3_layer7_op0)

    # agent3 - 4th layer [distribution center]
    agent3_layer4 = FulfillmentUnit()
    agent3_layer4_buffer0 = Buffer(resource=agent0_layer4_buffer0_resource, buffer_len=buffer_len)
    agent3_layer4.add_container(container=agent3_layer4_buffer0)
    agent3_layer4_op0 = OpRoute(buffers_orig=[agent3_layer4_buffer0], buffer_dest=agent3_layer7_buffer0, op_price=5.0,
                                op_time=2)
    agent3_layer4.add_operation(operation=agent3_layer4_op0)

    # agent3 - 3rd layer [collection warehouse]
    agent3_layer3 = FulfillmentUnit()
    agent3_layer3_buffer0 = Buffer(resource=agent0_layer3_buffer0_resource, buffer_len=buffer_len)
    agent3_layer3_buffer1 = Buffer(resource=agent0_layer3_buffer1_resource, buffer_len=buffer_len)
    agent3_layer3.add_container(container=agent3_layer3_buffer0)
    agent3_layer3.add_container(container=agent3_layer3_buffer1)
    agent3_layer3_op0 = OpRoute(buffers_orig=[agent3_layer3_buffer0, agent3_layer3_buffer1],
                                buffer_dest=agent3_layer4_buffer0, op_price=5.0, op_time=2)
    agent3_layer3.add_operation(operation=agent3_layer3_op0)

    # agent3 - 2nd layer [seller]
    agent3_layer2 = FulfillmentUnit()
    agent3_layer2_buffer0 = Buffer(resource=agent0_layer2_buffer0_resource, buffer_len=buffer_len)
    agent3_layer2.add_container(container=agent3_layer2_buffer0)
    agent3_layer2_op0 = OpRoute(buffers_orig=[agent3_layer2_buffer0], buffer_dest=agent3_layer3_buffer0, op_price=10.0,
                                op_time=1)
    agent3_layer2_op1 = OpRoute(buffers_orig=[agent3_layer2_buffer0], buffer_dest=agent3_layer3_buffer1, op_price=5.0,
                                op_time=2)
    agent3_layer2.add_operation(operation=agent3_layer2_op0)
    agent3_layer2.add_operation(operation=agent3_layer2_op1)

    # agent3 - 1st layer [e-commerce platform]
    agent3_layer1 = FulfillmentUnit()
    agent3_layer1_buffer0 = Buffer(resource=agent0_layer1_buffer0_resource, buffer_len=buffer_len)
    agent3_layer1.add_container(container=agent3_layer1_buffer0)
    agent3_layer1_op0 = OpRoute(buffers_orig=[agent3_layer1_buffer0], buffer_dest=agent3_layer2_buffer0, op_price=5.0,
                                op_time=1)
    agent3_layer1.add_operation(operation=agent3_layer1_op0)

    # agent3 - 0th layer [order placement]
    agent3_layer0 = FulfillmentUnit()
    agent3_layer0_order_queue = np.tile(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).astype(bool), 15)
    agent3_layer0_order_source = OrderSource(order_queue=agent3_layer0_order_queue)
    agent3_layer0_op0 = OpDispatch(order_src=agent3_layer0_order_source, buffer_dest=agent3_layer1_buffer0, op_price=0,
                                   op_time=1)
    agent3_layer0.add_operation(operation=agent3_layer0_op0)

    # agent3 assembling
    agent3 = Agent()
    agent3.add_fulfillment_unit(agent3_layer12)
    agent3.add_fulfillment_unit(agent3_layer11)
    agent3.add_fulfillment_unit(agent3_layer10)
    agent3.add_fulfillment_unit(agent3_layer9)
    agent3.add_fulfillment_unit(agent3_layer8)
    agent3.add_fulfillment_unit(agent3_layer7)
    agent3.add_fulfillment_unit(agent3_layer4)
    agent3.add_fulfillment_unit(agent3_layer3)
    agent3.add_fulfillment_unit(agent3_layer2)
    agent3.add_fulfillment_unit(agent3_layer1)
    agent3.add_fulfillment_unit(agent3_layer0)

    # ---------- AGENT 4 ---------- #
    # seller0-to-customer2 route0

    # agent4 - 12th layer [order delivered]
    agent4_layer12 = FulfillmentUnit()
    agent4_layer12_target_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent4_layer12_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent4_layer12_target = Inventory(resource=agent4_layer12_target_resource, inventory_limit=inventory_limit)
    agent4_layer12_buffer0 = Buffer(resource=agent4_layer12_buffer0_resource, buffer_len=buffer_len)
    agent4_layer12.add_container(container=agent4_layer12_target)
    agent4_layer12.add_container(container=agent4_layer12_buffer0)
    agent4_layer12_op0 = OpStore(buffers_orig=[agent4_layer12_buffer0], inventory_dest=agent4_layer12_target,
                                 op_price=0, op_time=0)
    agent4_layer12.add_operation(operation=agent4_layer12_op0)

    # agent4 - 9th layer [freight station arrival]
    agent4_layer9 = FulfillmentUnit()
    agent4_layer9_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent4_layer9_buffer0 = Buffer(resource=agent4_layer9_buffer0_resource, buffer_len=buffer_len)
    agent4_layer9.add_container(container=agent4_layer9_buffer0)
    agent4_layer9_op0 = OpRoute(buffers_orig=[agent4_layer9_buffer0], buffer_dest=agent4_layer12_buffer0, op_price=20.0,
                                op_time=4)
    agent4_layer9.add_operation(operation=agent4_layer9_op0)

    # agent4 - 8th layer [freight station departure]
    agent4_layer8 = FulfillmentUnit()
    agent4_layer8_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent4_layer8_buffer0 = Buffer(resource=agent4_layer8_buffer0_resource, buffer_len=buffer_len)
    agent4_layer8.add_container(container=agent4_layer8_buffer0)
    agent4_layer8_op0 = OpRoute(buffers_orig=[agent4_layer8_buffer0], buffer_dest=agent4_layer9_buffer0, op_price=40.0,
                                op_time=4)
    agent4_layer8.add_operation(operation=agent4_layer8_op0)

    # agent4 - 7th layer [linehaul warehouse]
    agent4_layer7 = FulfillmentUnit()
    agent4_layer7_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent4_layer7_buffer0 = Buffer(resource=agent4_layer7_buffer0_resource, buffer_len=buffer_len)
    agent4_layer7.add_container(container=agent4_layer7_buffer0)
    agent4_layer7_op0 = OpRoute(buffers_orig=[agent4_layer7_buffer0], buffer_dest=agent4_layer8_buffer0, op_price=5.0,
                                op_time=2)
    agent4_layer7.add_operation(operation=agent4_layer7_op0)

    # agent4 - 6th layer [distribution center of consolidation warehouse]
    agent4_layer6 = FulfillmentUnit()
    agent4_layer6_buffer0 = Buffer(resource=agent0_layer6_buffer0_resource, buffer_len=buffer_len)
    agent4_layer6.add_container(container=agent4_layer6_buffer0)
    agent4_layer6_op0 = OpRoute(buffers_orig=[agent4_layer6_buffer0], buffer_dest=agent4_layer7_buffer0, op_price=5.0,
                                op_time=2)
    agent4_layer6.add_operation(operation=agent4_layer6_op0)

    # agent4 - 5th layer [consolidation warehouse]
    agent4_layer5 = FulfillmentUnit()
    agent4_layer5_inventory = Inventory(resource=agent0_layer5_inventory_resource, inventory_limit=inventory_limit)
    agent4_layer5_buffer0 = Buffer(resource=agent0_layer5_buffer0_resource, buffer_len=buffer_len)
    agent4_layer5.add_container(container=agent4_layer5_inventory)
    agent4_layer5.add_container(container=agent4_layer5_buffer0)
    agent4_layer5_op0 = OpStore(buffers_orig=[agent4_layer5_buffer0], inventory_dest=agent4_layer5_inventory,
                                op_price=0.1, op_time=1)
    agent4_layer5_op1 = OpConsoRoute(buffers_orig=[agent4_layer5_buffer0], inventory_orig=agent4_layer5_inventory,
                                     buffer_dest=agent4_layer6_buffer0, op_price=5.0, op_time=2)
    agent4_layer5.add_operation(operation=agent4_layer5_op0)
    agent4_layer5.add_operation(operation=agent4_layer5_op1)

    # agent4 - 4th layer [distribution center]
    agent4_layer4 = FulfillmentUnit()
    agent4_layer4_buffer0 = Buffer(resource=agent0_layer4_buffer0_resource, buffer_len=buffer_len)
    agent4_layer4.add_container(container=agent4_layer4_buffer0)
    agent4_layer4_op0 = OpRoute(buffers_orig=[agent4_layer4_buffer0], buffer_dest=agent4_layer5_buffer0, op_price=5.0,
                                op_time=2)
    agent4_layer4.add_operation(operation=agent4_layer4_op0)

    # agent4 - 3rd layer [collection warehouse]
    agent4_layer3 = FulfillmentUnit()
    agent4_layer3_buffer0 = Buffer(resource=agent0_layer3_buffer0_resource, buffer_len=buffer_len)
    agent4_layer3_buffer1 = Buffer(resource=agent0_layer3_buffer1_resource, buffer_len=buffer_len)
    agent4_layer3.add_container(container=agent4_layer3_buffer0)
    agent4_layer3.add_container(container=agent4_layer3_buffer1)
    agent4_layer3_op0 = OpRoute(buffers_orig=[agent4_layer3_buffer0, agent4_layer3_buffer1],
                                buffer_dest=agent4_layer4_buffer0, op_price=5.0, op_time=2)
    agent4_layer3.add_operation(operation=agent4_layer3_op0)

    # agent4 - 2nd layer [seller]
    agent4_layer2 = FulfillmentUnit()
    agent4_layer2_buffer0 = Buffer(resource=agent0_layer2_buffer0_resource, buffer_len=buffer_len)
    agent4_layer2.add_container(container=agent4_layer2_buffer0)
    agent4_layer2_op0 = OpRoute(buffers_orig=[agent4_layer2_buffer0], buffer_dest=agent4_layer3_buffer0, op_price=10.0,
                                op_time=1)
    agent4_layer2_op1 = OpRoute(buffers_orig=[agent4_layer2_buffer0], buffer_dest=agent4_layer3_buffer1, op_price=5.0,
                                op_time=2)
    agent4_layer2.add_operation(operation=agent4_layer2_op0)
    agent4_layer2.add_operation(operation=agent4_layer2_op1)

    # agent4 - 1st layer [e-commerce platform]
    agent4_layer1 = FulfillmentUnit()
    agent4_layer1_buffer0 = Buffer(resource=agent0_layer1_buffer0_resource, buffer_len=buffer_len)
    agent4_layer1.add_container(container=agent4_layer1_buffer0)
    agent4_layer1_op0 = OpRoute(buffers_orig=[agent4_layer1_buffer0], buffer_dest=agent4_layer2_buffer0, op_price=5.0,
                                op_time=1)
    agent4_layer1.add_operation(operation=agent4_layer1_op0)

    # agent4 - 0th layer [order placement]
    agent4_layer0 = FulfillmentUnit()
    agent4_layer0_order_queue = np.tile(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).astype(bool), 15)
    agent4_layer0_order_source = OrderSource(order_queue=agent4_layer0_order_queue)
    agent4_layer0_op0 = OpDispatch(order_src=agent4_layer0_order_source, buffer_dest=agent4_layer1_buffer0, op_price=0,
                                   op_time=1)
    agent4_layer0.add_operation(operation=agent4_layer0_op0)

    # agent4 assembling
    agent4 = Agent()
    agent4.add_fulfillment_unit(agent4_layer12)
    agent4.add_fulfillment_unit(agent4_layer9)
    agent4.add_fulfillment_unit(agent4_layer8)
    agent4.add_fulfillment_unit(agent4_layer7)
    agent4.add_fulfillment_unit(agent4_layer6)
    agent4.add_fulfillment_unit(agent4_layer5)
    agent4.add_fulfillment_unit(agent4_layer4)
    agent4.add_fulfillment_unit(agent4_layer3)
    agent4.add_fulfillment_unit(agent4_layer2)
    agent4.add_fulfillment_unit(agent4_layer1)
    agent4.add_fulfillment_unit(agent4_layer0)

    # ---------- AGENT 5 ---------- #
    # seller0-to-customer2 route1

    # agent5 - 12th layer [order delivered]
    agent5_layer12 = FulfillmentUnit()
    agent5_layer12_target = Inventory(resource=agent4_layer12_target_resource, inventory_limit=inventory_limit)
    agent5_layer12_buffer0 = Buffer(resource=agent4_layer12_buffer0_resource, buffer_len=buffer_len)
    agent5_layer12.add_container(container=agent5_layer12_target)
    agent5_layer12.add_container(container=agent5_layer12_buffer0)
    agent5_layer12_op0 = OpStore(buffers_orig=[agent5_layer12_buffer0], inventory_dest=agent5_layer12_target,
                                 op_price=0, op_time=0)
    agent5_layer12.add_operation(operation=agent5_layer12_op0)

    # agent5 - 9th layer [freight station arrival]
    agent5_layer9 = FulfillmentUnit()
    agent5_layer9_buffer0 = Buffer(resource=agent4_layer9_buffer0_resource, buffer_len=buffer_len)
    agent5_layer9.add_container(container=agent5_layer9_buffer0)
    agent5_layer9_op0 = OpRoute(buffers_orig=[agent5_layer9_buffer0], buffer_dest=agent5_layer12_buffer0, op_price=20.0,
                                op_time=4)
    agent5_layer9.add_operation(operation=agent5_layer9_op0)

    # agent5 - 8th layer [freight station departure]
    agent5_layer8 = FulfillmentUnit()
    agent5_layer8_buffer0 = Buffer(resource=agent4_layer8_buffer0_resource, buffer_len=buffer_len)
    agent5_layer8.add_container(container=agent5_layer8_buffer0)
    agent5_layer8_op0 = OpRoute(buffers_orig=[agent5_layer8_buffer0], buffer_dest=agent5_layer9_buffer0, op_price=4.0,
                                op_time=4)
    agent5_layer8.add_operation(operation=agent5_layer8_op0)

    # agent5 - 7th layer [linehaul warehouse]
    agent5_layer7 = FulfillmentUnit()
    agent5_layer7_buffer0 = Buffer(resource=agent4_layer7_buffer0_resource, buffer_len=buffer_len)
    agent5_layer7.add_container(container=agent5_layer7_buffer0)
    agent5_layer7_op0 = OpRoute(buffers_orig=[agent5_layer7_buffer0], buffer_dest=agent5_layer8_buffer0, op_price=5.0,
                                op_time=2)
    agent5_layer7.add_operation(operation=agent5_layer7_op0)

    # agent5 - 4th layer [distribution center]
    agent5_layer4 = FulfillmentUnit()
    agent5_layer4_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent5_layer4_buffer0 = Buffer(resource=agent5_layer4_buffer0_resource, buffer_len=buffer_len)
    agent5_layer4.add_container(container=agent5_layer4_buffer0)
    agent5_layer4_op0 = OpRoute(buffers_orig=[agent5_layer4_buffer0], buffer_dest=agent5_layer7_buffer0, op_price=10.0,
                                op_time=3)
    agent5_layer4.add_operation(operation=agent5_layer4_op0)

    # agent5 - 3rd layer [collection warehouse]
    agent5_layer3 = FulfillmentUnit()
    agent5_layer3_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent5_layer3_buffer1_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent5_layer3_buffer0 = Buffer(resource=agent5_layer3_buffer0_resource, buffer_len=buffer_len)
    agent5_layer3_buffer1 = Buffer(resource=agent5_layer3_buffer1_resource, buffer_len=buffer_len)
    agent5_layer3.add_container(container=agent5_layer3_buffer0)
    agent5_layer3.add_container(container=agent5_layer3_buffer1)
    agent5_layer3_op0 = OpRoute(buffers_orig=[agent5_layer3_buffer0, agent5_layer3_buffer1],
                                buffer_dest=agent5_layer4_buffer0, op_price=5.0, op_time=2)
    agent5_layer3.add_operation(operation=agent5_layer3_op0)

    # agent5 - 2nd layer [seller]
    agent5_layer2 = FulfillmentUnit()
    agent5_layer2_buffer0_resource = Resource(constraint=-1, normal_price=0.0, overage_price=0.0, occupied=0)
    agent5_layer2_buffer0 = Buffer(resource=agent5_layer2_buffer0_resource, buffer_len=buffer_len)
    agent5_layer2.add_container(container=agent5_layer2_buffer0)
    agent5_layer2_op0 = OpRoute(buffers_orig=[agent5_layer2_buffer0], buffer_dest=agent5_layer3_buffer0, op_price=8.0,
                                op_time=1)
    agent5_layer2_op1 = OpRoute(buffers_orig=[agent5_layer2_buffer0], buffer_dest=agent5_layer3_buffer1, op_price=4.0,
                                op_time=2)
    agent5_layer2.add_operation(operation=agent5_layer2_op0)
    agent5_layer2.add_operation(operation=agent5_layer2_op1)

    # agent5 - 1st layer [e-commerce platform]
    agent5_layer1 = FulfillmentUnit()
    agent5_layer1_buffer0 = Buffer(resource=agent0_layer1_buffer0_resource, buffer_len=buffer_len)
    agent5_layer1.add_container(container=agent5_layer1_buffer0)
    agent5_layer1_op0 = OpRoute(buffers_orig=[agent5_layer1_buffer0], buffer_dest=agent5_layer2_buffer0, op_price=5.0,
                                op_time=1)
    agent5_layer1.add_operation(operation=agent5_layer1_op0)

    # agent5 - 0th layer [order placement]
    agent5_layer0 = FulfillmentUnit()
    agent5_layer0_order_queue = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).astype(bool), 15)
    agent5_layer0_order_source = OrderSource(order_queue=agent5_layer0_order_queue)
    agent5_layer0_op0 = OpDispatch(order_src=agent5_layer0_order_source, buffer_dest=agent5_layer1_buffer0, op_price=0,
                                   op_time=1)
    agent5_layer0.add_operation(operation=agent5_layer0_op0)

    # agent5 assembling
    agent5 = Agent()
    agent5.add_fulfillment_unit(agent5_layer12)
    agent5.add_fulfillment_unit(agent5_layer9)
    agent5.add_fulfillment_unit(agent5_layer8)
    agent5.add_fulfillment_unit(agent5_layer7)
    agent5.add_fulfillment_unit(agent5_layer4)
    agent5.add_fulfillment_unit(agent5_layer3)
    agent5.add_fulfillment_unit(agent5_layer2)
    agent5.add_fulfillment_unit(agent5_layer1)
    agent5.add_fulfillment_unit(agent5_layer0)

    return [agent0, agent1, agent2, agent3, agent4, agent5]
