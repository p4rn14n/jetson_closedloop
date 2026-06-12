from audiostream import get_output
from audiostream.sources.wave import SineSource
import time


freqs = [1000 * (2**(1/4))**i for i in range(18)]
austream = get_output(channels=1, rate=44100, buffersize=1024)

for freq in freqs:
    sinsource = SineSource(austream, freq)
    sinsource.start()
    time.sleep(1)
    sinsource.stop()