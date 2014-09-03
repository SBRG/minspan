try:
    from setuptools import setup
except:
    from distutils import setup

setup(
    name="minspan",
    version="0.1.0",
    py_modules=["minspan"],
    install_requires=["cobra"],
    author="Ali Ebrahim and Aarash Bordbar",
    author_email="aebrahim@ucsd.edu",
    url="https://github.com/SBRG/minspan",
    license="MIT",
    classifiers=["License :: OSI Approved :: MIT License",
                 "Intended Audience :: Science/Research",
                 ],
)
