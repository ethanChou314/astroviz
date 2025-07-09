from setuptools import setup, find_packages

setup(
    name='astroviz',
    version='0.1.0',
    description='A tool for data analysis and visualization in radio astronomy',
    author='Ethan Chou',
    author_email='ethanchou04@gmail.com',
    packages=find_packages(),  # this finds the 'astroviz/' package automatically
    install_requires=[
        'numpy',
        'pandas',
        'astropy',
        'scipy',
        'matplotlib',
        'astroquery',
    ],
    python_requires='>=3.8',
)