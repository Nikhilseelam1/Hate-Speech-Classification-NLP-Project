from hate.logger import logging
from hate.exception import CustomException

try:
    a=5/'0'
except Exception as e:
    raise CustomException(e) from e