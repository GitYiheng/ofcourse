from env.order import Order
from env.order_source import OrderSource


def test_order_update_accepts_zero_values():
    order = Order()
    order.update(specified_price=0, specified_time=0)

    assert order.cumulative_price == 0
    assert order.cumulative_time == 0


def test_order_source_accepts_missing_queue():
    source = OrderSource(order_queue=None, order_queue_len=5)

    assert len(source.order_queue) == 5
    assert source.num_order >= 0
