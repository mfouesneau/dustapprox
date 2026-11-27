import matplotlib.pyplot as plt
from pkg_resources import resource_filename
from pyphot.libraries import Ascii_Library

where = resource_filename('dustapprox', 'data/Gaia2')
lib = Ascii_Library(where, glob_pattern='*csv')
pbs = lib.load_all_filters()

plt.figure(figsize=(8, 4))
for pb in pbs:
    if pb.name.startswith('C1B'):
        kwargs = dict(color='k', lw=2)
    else:
        kwargs = dict(color='C0', lw=1)
    plt.plot(pb.wavelength.to('nm'), pb.transmit, **kwargs)
plt.xlabel('wavelength [nm]')
plt.ylabel('throughput')
plt.tight_layout()
plt.show()