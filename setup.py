from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='wwgd',
      version='0.0.1',
      description='Image caption generator',
      long_description="Generate captions for images using standard deep \
                        learning based image processing solutions",
      classifiers=[
                   'Development Status :: 1 - Planning',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: Unix',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                  ],
      keywords='image processing DL deep learning',
      url='https://github.com/inJeans/what-would-gordon-do',
      author='Christopher Jon Watkins',
      author_email='christopher.watkins@me.com',
      license='MIT',
      packages=['wwgd'],
      install_requires=[
                        'markdown',
                       ],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      entry_points = {
                      'console_scripts': ['wwgd-cli=wwgd.command_line:main'],
                     },
      zip_safe=False)
