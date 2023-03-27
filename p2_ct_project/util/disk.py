import gzip

from diskcache import FanoutCache, Disk
from diskcache.core import BytesType, MODE_BINARY, BytesIO

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

