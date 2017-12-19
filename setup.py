from setuptools import setup, find_packages
setup(
    name='future',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        ('/etc/future/', ['future.conf'])
    ],
    install_requires=[],
    scripts=[
        'bin/stock_update',
        'bin/stock_predict',
        'bin/gen_train_set',
        'bin/stock_realtime'
    ],
)
