from distutils.core import setup

setup(
    name='hdf5tools',
    version='0.1.0',
    author='Benjamin Farmer',
    author_email='ben.farmer@gmail.com',
    packages=['hdf5tools'],
    scripts=['bin/addtohdf5_xe15k.py','bin/evidence.py','bin/newevidences.py',\
'bin/quickinfo.py','bin/reweight.py','bin/summary.py','bin/treehdf5.py'],
    url='http://pypi.python.org/pypi/hdf5tools/', #not really, just an example
    license='LICENSE.txt',
    description='A collection of analysis tools for performing Bayesian model \
comparison and parameter estimation, on data produced in MultiNest (PySUSY 2&3) \
format',
    long_description=open('README.txt').read(),
#    install_requires=[
#        "MultiNest == 2.12",
#    ],
)
