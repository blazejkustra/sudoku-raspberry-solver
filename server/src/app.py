from flask import Flask, request
from config import get_config
from utils.utils import get_env_variable
from predict import predict_number


def create_app():
    app = Flask(__name__)
    app.config.from_object(get_config())

    @app.route('/predict', methods=['POST'])
    def predict():
        return predict_number(request.data)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run()
