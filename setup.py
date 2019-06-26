from setuptools import setup

setup(
    name='shapSD',
    version='0.1.1',
    packages=['shapSD', 'shapSD.feature_explainer', 'shapSD.pysubgroup'],
    package_dir={'shapSD': 'shapSD'},
    url='https://github.com/XiaoqiMa/shapSD',
    download_url='https://github.com/XiaoqiMa/shapSD/archive/V0.1-alpha.tar.gz',
    license='MIT License (MIT)',
    author='xiaoqi',
    author_email='xiaoqima2013@gmail.com',
    description='Explain variable influence with shapley values in black box model through pattern mining',
    install_requires=[
        'pandas', 'scipy', 'numpy', 'matplotlib', 'eli5', 'shap',
        'scikit-learn', 'lightgbm', 'graphviz'
    ],
    python_requires='>=3.6'
)
