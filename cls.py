from abc import ABC, abstractmethod
import yaml

class ModelLoader(ABC):
    @abstractmethod
    def load_model(self):
        pass

    def load_options(self, path):
        model_param = None
        if path is not None:
            with open(path, 'r') as f:
                model_param = yaml.safe_load(f)
        return model_param

    def out_opts(self):
        pass

    def serial_opts(self, options):
        return ' ,'.join(f"{k}: {v}"
                             for k, v in vars(options).items()
                             if not (k.startswith('__')))
