from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('gdr3_dustapprox/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name = "gdr3_dustapprox",
    version = main_ns['__VERSION__'],
    description = "A tool for computing extinction coefficients in a quick and dirty manner",
    long_description = readme(),
    author = "Morgan Fouesneau",
    author_email = "",
    url = "https://github.com/mfouesneau/gdr3_dustapprox",
    packages = find_packages(),
    package_data = {},
    include_package_data = True,
    classifiers=[
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: MIT License',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=["numpy", "scipy"]
)
