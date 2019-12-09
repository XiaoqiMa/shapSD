from setuptools import setup

setup(
    name='shapSD',
    version='0.3.1',
    packages=['shapSD', 'shapSD.feature_explainer', 'shapSD.pysubgroup'],
    package_dir={'shapSD': 'shapSD', 'shapSD.feature_explainer': 'shapSD/feature_explainer',
                 'shapSD.pysubgroup': 'shapSD/pysubgroup'},
    url='https://github.com/XiaoqiMa/shapSD',
    download_url='https://github.com/XiaoqiMa/shapSD/archive/v2.0.tar.gz',
    license='MIT License (MIT)',
    author='xiaoqi',
    author_email='xiaoqima2013@gmail.com',
    description='An black-box interpretation framework to explain variable influence',
    install_requires=[
        'pandas', 'scipy', 'numpy', 'matplotlib', 'eli5', 'shap',
        'scikit-learn', 'lightgbm', 'graphviz'
    ],
    python_requires='>=3.6'
)
