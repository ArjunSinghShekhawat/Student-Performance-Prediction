from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requitements(filename:str)->List[str]:
    
    """This function return the list of requirements"""

    requirements = []
    with open(filename) as file_obj:
        requirements = file_obj.readline()
        requirements = [req.replace('\n',"") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements





setup(
    name="Student Performance Prediction",
    author='Arjun Singh',
    author_email="shekhawatsingharjun12345@gmail.com",
    packages=find_packages(),
    install_requires = get_requitements('requirements.txt')
)