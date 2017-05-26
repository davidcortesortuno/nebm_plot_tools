from setuptools import setup

setup(
    name="nebm_plot_tools",
    version="1.1",
    author="D. I. Cortes",
    description='Plotting library for NEBM data based on Matplotlib and Mayavi',
    url='https://github.com/davidcortesortuno/nebm_plot_tools',
    install_requires=[
        'matplotlib', 'mayavi', 'numpy'
    ],
    packages=['nebm_plot_tools'],
    license='MIT',
)
