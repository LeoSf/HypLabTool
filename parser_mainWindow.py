#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biong. Leandro D. Medus
Ph.D Student GPDD - ETSE
Universitat de València
leandro.d.medus@uv.es

23-04-2019

Parser to fix the automatic generation of the script from Qt Designer to the python
file.

"""
__author__ = "Leandro D. Medus <leandro.d.medus@uv.es>"
__version__ = '1.0.0'


if __name__ == "__main__":
    import re

    regex = re.compile(r'= QtImageViewer\(self\.tab_preview\)')
    string_substitution = "= QtImageViewer()"

    file_mainWindow = "ui/mainWindow.py"

    in_file = open(file_mainWindow, "rt")
    contents = in_file.read()
    in_file.close()

    contents = re.sub(regex, string_substitution, contents)

    fd = open(file_mainWindow, "w+")

    fd.write(contents)

    fd.close()
