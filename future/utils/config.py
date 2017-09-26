import configparser
config = configparser.RawConfigParser()
config.read('/etc/future/future.conf')

def get_config(section, option, default = None):
    try:
        return config.get(section, option)
    except Exception as e:
        return default
