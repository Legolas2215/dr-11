import logging
from datetime import datetime

class SingletonMeta(type):
    """
    A Singleton metaclass that ensures only one instance of the Logger class exists.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Logger(metaclass=SingletonMeta):
    """
    A Singleton Logger class that logs messages in a specific format.
    Format: Datetime | log level | status code | function name | log message
    """

    def __init__(self):
        self.logger = logging.getLogger("D11Logger")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(status_code)s | %(funcName)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log(self, level, status_code, func_name, message):
        """
        Logs a message with the given level, status code, function name, and message.

        :param level: The log level (e.g., logging.INFO, logging.ERROR)
        :param status_code: The status code associated with the log message
        :param func_name: The name of the function where the log is being made
        :param message: The log message
        """
        extra = {'status_code': status_code, 'funcName': func_name}
        self.logger.log(level, message, extra=extra)
