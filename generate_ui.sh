#!/bin/bash

pyuic5 -x ui/mainWindow.ui -o ui/mainWindow.py

python3 parser_mainWindow.py