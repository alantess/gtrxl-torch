from setuptools import setup,find_packages

setup(
        name="gtrxl-torch",
        version = '0.1.5',
        license="MIT", 
        description = "Gated-Transformer XL - PyTorch",
        author = "Alan Tessier",
        author_email = "alantessier97@gmail.com",
        url = "https://github.com/alantess/gtrxl-torch",
        keywords = ["transformer", "computer vision", "deep learning", "artifical intelligence"],
        install_requires = ["torch>=1.6"],
        classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        ],
        packages = find_packages(exclude=["examples"])

        )
