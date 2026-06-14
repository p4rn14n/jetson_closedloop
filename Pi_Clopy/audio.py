from audiostream import get_output
from audiostream.sources.wave import SineSource
import time
stream = get_output(channels = 1, rate=44100, buffersize=128)
# sinsource = SineSource(stream, 400 * (2**(1/4))**1)
n_tones = 18
freqs = [1000 * (2**(1/4))**i for i in range(n_tones)]

for freq in freqs:
    sinsource = SineSource(stream, freq)
    sinsource.start()
    time.sleep(1)
    sinsource.stop()
#sinsource = SineSource(stream, 1000* (2**(1/4))**1)




# sinsource = SineSource(stream, 400 * (2**(1/4))**18)
# sinsource.start()
# time.sleep(3)
# sinsource.stop()

# time.sleep(1)

# sinsource = SineSource(stream, 1000 * (2**(1/4))**1)
# sinsource.start()
# time.sleep(3)
# sinsource.stop()


# sinsource = SineSource(stream, 1000 * (2**(1/4))**16)
# sinsource.start()
# time.sleep(3)
# sinsource.stop()
# stream = get_output(channels = 1, rate=22050, buffersize=128)
# sinsource = SineSource(stream, 1000)
# sinsource.start()
# time.sleep(3)
# sinsource.stop()

# time.sleep(1)

# stream = get_output(channels = 1, rate=44100, buffersize=256)
# sinsource = SineSource(stream, 1000)
# sinsource.start()
# time.sleep(3)
# sinsource.stop()

# time.sleep(1)

# stream = get_output(channels = 1, rate=44100, buffersize=64)
# sinsource = SineSource(stream, 1000)
# sinsource.start()
# time.sleep(3)
# sinsource.stop()