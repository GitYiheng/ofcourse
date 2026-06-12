import threading

import requests


class VisualizerClient:
    def __init__(self, server_url="http://localhost:5000/api/state", timeout=0.05):
        self.server_url = server_url
        self.timeout = timeout

    def send(self, state):
        thread = threading.Thread(target=self._send, args=(state,), daemon=True)
        thread.start()

    def _send(self, state):
        try:
            requests.post(self.server_url, json=state, timeout=self.timeout)
        except requests.exceptions.RequestException:
            pass
