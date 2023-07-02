import logging
import sys
logging.basicConfig(format='%(filename)s:%(lineno)s | %(levelname)s : %(message)s',
                     level=logging.WARNING, stream=sys.stderr)