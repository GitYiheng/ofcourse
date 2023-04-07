import logging
import os.path
import time


class Logger:
    def __init__(self, args):
        self.logger = None
        self.path = self.get_log_path(args)

    def get_logger(self, format=None):
        self.logger = logging.getLogger(self.path)
        self.logger.setLevel(logging.INFO)

        if format is None:
            fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        else:
            fmt = logging.Formatter('')

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)

        fh = logging.FileHandler(self.path)
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
        return self.logger

    def get_log_path(self, args):
        log_info_dir = "logs"
        if not os.path.exists(log_info_dir):
            os.makedirs(log_info_dir)
        return "{}/log_{}_{}_seed{}_{}.txt".format(log_info_dir, args.env, args.algo, args.seed, int(time.time()))

    def print_logo(self):
        self.logger.info('''
            
          /$$$$$$  /$$$$$$$$ /$$$$$$   /$$$$$$  /$$   /$$ /$$$$$$$   /$$$$$$  /$$$$$$$$
         /$$__  $$| $$_____//$$__  $$ /$$__  $$| $$  | $$| $$__  $$ /$$__  $$| $$_____/
        | $$  \ $$| $$     | $$  \__/| $$  \ $$| $$  | $$| $$  \ $$| $$  \__/| $$      
        | $$  | $$| $$$$$  | $$      | $$  | $$| $$  | $$| $$$$$$$/|  $$$$$$ | $$$$$   
        | $$  | $$| $$__/  | $$      | $$  | $$| $$  | $$| $$__  $$ \____  $$| $$__/   
        | $$  | $$| $$     | $$    $$| $$  | $$| $$  | $$| $$  \ $$ /$$  \ $$| $$      
        |  $$$$$$/| $$     |  $$$$$$/|  $$$$$$/|  $$$$$$/| $$  | $$|  $$$$$$/| $$$$$$$$
         \______/ |__/      \______/  \______/  \______/ |__/  |__/ \______/ |________/
                                                                            

        ''')
