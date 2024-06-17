@echo off
REM ####################################################################################################
REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License. See LICENSE in the project root for license information.
REM Build script for the Assera Python package
REM ####################################################################################################

pip install -r requirements.txt
copy ../../docs/assets/logos/Assera_darktext.svg static/.

REM python viz_tool.py [--port <port>]
python viz_tool.py