"""
Stable Diffusion 图片生成器安装脚本
"""

from setuptools import setup, find_packages
import sys
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Stable Diffusion 图片生成器"

# 读取requirements文件
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="sd-image-generator",
    version="1.0.0",
    description="基于Stable Diffusion的AI图片生成器",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="AI Assistant",
    author_email="ai@example.com",
    url="https://github.com/example/sd-image-generator",
    
    packages=find_packages(),
    py_modules=[
        'main',
        'gui',
        'sd_generator',
        'config',
        'utils',
        'start'
    ],
    
    install_requires=read_requirements(),
    
    python_requires=">=3.8",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    keywords="stable-diffusion ai image-generation deep-learning pytorch",
    
    entry_points={
        'console_scripts': [
            'sd-generator=main:main',
            'sd-start=start:main',
        ],
    },
    
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.yml', '*.yaml'],
    },
    
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=3.8',
        ],
        'gpu': [
            'torch[cuda]',
        ],
    },
    
    project_urls={
        'Bug Reports': 'https://github.com/example/sd-image-generator/issues',
        'Source': 'https://github.com/example/sd-image-generator',
        'Documentation': 'https://github.com/example/sd-image-generator/wiki',
    },
)
