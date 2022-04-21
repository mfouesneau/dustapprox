from setuptools import setup, find_packages
from distutils.util import convert_path

def version():
    """Get the version tag from the package """
    main_ns = {}
    ver_path = convert_path('dustapprox/version.py')
    with open(ver_path) as ver_file:
        exec(ver_file.read(), main_ns)
    return main_ns['__VERSION__']


def readme():
    """ get readme content """
    with open('README.rst') as f:
        return f.read()


def requirements():
    """ get requirements content """
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(name = "dustapprox",
    version = version(),
    description = "A tool for computing extinction coefficients in a quick and dirty manner",
    long_description = readme(),
    author = "Morgan Fouesneau",
    author_email = "",
    url = "https://github.com/mfouesneau/dustapprox",
    packages = find_packages(),
    package_data = {'dustapprox':['data/*']},
    include_package_data = True,
    classifiers=[
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: BSD License',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=requirements()
)
