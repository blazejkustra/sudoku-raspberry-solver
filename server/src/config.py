import os
from os.path import join, dirname
from dotenv import load_dotenv
from utils.utils import get_env_variable

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


class Config(object):
    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    DEBUG = True


class TestConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    pass  # TODO


def get_config():
    try:
        FLASK_ENV = get_env_variable('FLASK_ENV')
    except Exception:
        FLASK_ENV = 'development'
        print('FLASK_ENV is not set, using FLASK_ENV:', FLASK_ENV)

    if FLASK_ENV == 'production':
        return ProductionConfig()
    if FLASK_ENV == 'test':
        return TestConfig()

    return DevelopmentConfig()
