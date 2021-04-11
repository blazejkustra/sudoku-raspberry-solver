import os
from flask_script import Manager
from tests.predict_test import predict_test

from app import create_app

app = create_app()

manager = Manager(app=app)


@manager.command
def run():
    app.run()


@manager.command
def test():
    predict_test()
    print("All tests has passed")


if __name__ == '__main__':
    manager.run()
