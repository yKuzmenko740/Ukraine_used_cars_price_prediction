import yaml


class Config:
    def __init__(self, data: dict):
        self.main_data = data['data_path']['main_csv']
        self.model_coefs = data['model']['model_coefficients']

    @staticmethod
    def load_from_file() -> "Config":
        """
        Reads config from the default YAML file
        """
        with open("config.yml") as f:
            data = yaml.safe_load(f)
        return Config(data)