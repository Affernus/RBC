from setuptools import find_packages, setup

setup(
    name='rbk',
    version='1.0',
    description='module for hacks-ai.ru RBK championship',
    author='Affernus',
    author_email='n_moshkov@mail.ru',
    packages=find_packages('src'),
    package_dir={'': 'src'}
)