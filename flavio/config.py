import yaml
import pkgutil

f = pkgutil.get_data('flavio', 'data/config.yml')
config = yaml.safe_load(f)
