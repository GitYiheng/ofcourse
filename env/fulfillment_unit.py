from collections import deque


class FulfillmentUnit:
    def __init__(self, latitude=None, longitude=None):
        self.containers = deque()
        self.operations = deque()

        self._set_coordinates(latitude, longitude)  # geographic information

    def _set_coordinates(self, latitude=None, longitude=None):
        self.latitude = None
        self.longitude = None

    def add_container(self, container=None):
        if container: self.containers.append(container)

    def add_operation(self, operation=None):
        if operation: self.operations.append(operation)

    def update(self):
        _step_price = 0
        for _container in self.containers:
            _step_price += _container.update()
        return _step_price

    def execute(self, action, step_count=None):
        return self.operations[action].execute(step_count)

    def get_obs(self):
        _obs = []
        for _container in self.containers:
            _obs += _container.get_obs()
        for _operation in self.operations:
            _obs += _operation.get_obs()
        return _obs
