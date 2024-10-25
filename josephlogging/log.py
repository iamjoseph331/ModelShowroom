import logging
import logging.config
from config.config import cfg, logging_level

# to add customer log filed that are request_id and log_type, we sub class logger
# not using customer logger factory because we can not pass extra to it, and the default logger will
# raise exception when the extra has conflict attribute check.

class CustomLogger(logging.Logger):
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func = None, extra = None, sinfo = None):

        rv = super().makeRecord(name, level, fn, lno, msg, args, exc_info, func,
            extra, sinfo)

        if extra is not None:
            for key in extra:
                # if (key in ["message", "asctime"]) or (key in rv.__dict__):
                # we removed the conflict attribute check from the superclass
                if key in ["message", "asctime"]:
                    raise KeyError("Attempt to overwrite %r in LogRecord" % key)
                rv.__dict__[key] = extra[key]

        return rv

res = cfg.get('logging')
if logging_level != 'Default':
    res['root']['level'] =logging_level
logging.getLogger('matplotlib').disabled = True
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
logging.config.dictConfig(res)
logging.setLoggerClass(CustomLogger)  # use the subclassed logger


def getLogger(name):
    return logging.getLogger(name)  # just use this as a wrapper for logging module

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.warning("this is a warning", extra = {'log_type': 'A', "request_id": "98934"})
    logger.error("hello", extra = {"request_id": "10000"})
    logger.debug("world", extra = {"request_id": "55555"})
    logger.debug("next", extra = {"log_type": "D"})
    logger.debug("nothing")