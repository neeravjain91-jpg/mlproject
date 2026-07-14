from setuptools  import find_packages,setup
from typing import List
HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str)->list[str]:
    '''this function will return the list of requirements'''
requirements=[]
with open('requirements.txt') as file_obj:
    requirements=file_obj.readlines()
    requirements=[req.replace("/n","") for req in requirements]
    if HYPHEN_E_DOT  in requirements:
        requirements.remove(HYPHEN_E_DOT)
        setup(
    name='mlproject',
    version='0.0.1',
    author='neerav',
    author_email='neeravjain91@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirments.txt')

)