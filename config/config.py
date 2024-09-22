import sys
import os
import yaml

def get_conf(config_path):
    try:
        with open(config_path) as f:
            res = yaml.safe_load(f)
            return res
    except Exception:
        sys.stderr.write("error: failed to open config file\n")

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    conf_path = os.environ["CONF_PATH"]  # k8s can use this environment to specify the config file path (say config map)
except KeyError:
    conf_path = dir_path + "/config.yaml"  # if not defined in env, use the default config

cfg = get_conf(conf_path)
try:
    public_host = os.environ['PUBLIC_HOST']
except KeyError:
    public_host = 'https://JosephPlatform.herokuapp.com'

try:
    internalmodels_key = os.environ['INTERNAL_KEY']
except KeyError:
    internalmodels_key = 'DefaultKey'

try:
    logging_level = os.environ['LOGGING_LEVEL']
except KeyError:
    logging_level = 'Default'

try:
    port = int(os.environ['PORT'])  # heroku will set this
except KeyError:    
    port = cfg['port']
cfg['port'] = port

if __name__ == "__main__":
    print(cfg)
