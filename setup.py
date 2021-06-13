from setuptools import find_packages, setup

__version__ = "0.0.1"

if __name__ == '__main__':
    setup(
        name='torch_speaker',
        version=__version__,
        description='Deep Learning Speaker Recognition Toolbox',
        url='',
        author='Yang Zhang',
        author_email='zyziszy@foxmail.com',
        keywords='speaker recognition',
        packages=find_packages(exclude=('config', 'tools', 'scripts')),
        classifiers=[
            'Development Status :: Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        zip_safe=False)
