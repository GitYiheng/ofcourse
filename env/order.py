from env.utils import ServiceLevel


class Order:
    def __init__(self, reference_number=None, service_level=ServiceLevel.standard):
        # current order-wise cost
        self.cumulative_price = 0
        self.cumulative_time = 0

        # cost per time step
        self.recurring_price = 0
        self.recurring_time = 1

        # max time for selected service level
        self.max_price = None
        self.max_time = None

        self._assign_reference_number(reference_number)
        self.select_service_level(service_level)

    def select_service_level(self, service_level):
        self.service_level = service_level
        # payment and lead-time for chosen service level
        if service_level == ServiceLevel.premium: self.max_price = 50; self.max_time = 10
        if service_level == ServiceLevel.standard: self.max_price = 40; self.max_time = 20
        if service_level == ServiceLevel.economy: self.max_price = 30; self.max_time = 30

    def set_recurring_price(self, recurring_price=None):
        if recurring_price: self.recurring_price = recurring_price

    def update(self, specified_price=None, specified_time=None):
        _step_price = specified_price if specified_price else self.recurring_price
        self.cumulative_price += _step_price
        self.cumulative_time += specified_time if specified_time else self.recurring_time
        return _step_price

    def get_new(self):
        return self.__class__()

    def _assign_reference_number(self, reference_number):
        self.reference_number = reference_number if reference_number else id(self)

    def get_obs(self):
        return [self.cumulative_price, self.cumulative_time, self.recurring_price, self.recurring_time, self.max_price,
                self.max_time]
