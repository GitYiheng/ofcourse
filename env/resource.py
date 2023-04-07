class Resource:
    def __init__(self, constraint=-1, normal_price=0.1, overage_price=0.5, occupied=0):
        # constraint=0 for unavailable resource, always use overage_price
        # constraint=-1 for unlimited resource, always use normal_price
        self.occupied = occupied  # used resource
        self.constraint = constraint  # capacity
        self.normal_price = normal_price  # normal cost per parcel
        self.overage_price = overage_price  # overage cost per parcel

    def occupy(self, occupied_size):
        self.occupied += occupied_size

    def release(self, released_size):
        self.occupied -= released_size

    def _is_resource_sufficient(self):
        return self.occupied < self.constraint or self.constraint == -1

    def get_recurring_price(self):
        # obtain recurring price based on resource status
        return self.normal_price if self._is_resource_sufficient() else self.overage_price

    def get_obs(self):
        _obs = [self.occupied, self.constraint, self.normal_price, self.overage_price]
        return _obs
