import matplotlib.pyplot as plt

from dustapprox.io import svo
which_filters = ['GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp', 'GAIA/GAIA3.G']
passbands = svo.get_svo_passbands(which_filters)

for pb in passbands:
   plt.plot(pb.wavelength.to('nm'), pb.transmit, label=pb.name)

plt.legend(loc='upper right', frameon=False)

plt.xlabel('wavelength [nm]')
plt.ylabel('transmission')
plt.tight_layout()
plt.show()