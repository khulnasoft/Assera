#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Authors:  Mason Remy
# Requires: Python 3.7+, tomlkit
#
# Utility to parse the TOML metadata from Assera headers
####################################################################################################

import argparse
import os

# Requires tomlkit: pip install tomlkit
import tomlkit


class AsseraModuleData:
    def __init__(self, toml_table):
        self.module_toml = toml_table
        self.name = toml_table["module_name"]
        self.metadata = toml_table["metadata"]
        self.code_table = toml_table["code"]
        variant_metadata_keys = ["_function", "_initialize_function", "_deinitialize_function", "domain"]
        self.is_assera_variant = True
        for key in variant_metadata_keys:
            if key not in self.metadata:
                # Utility modules have TOML data, but don't have Assera sample variant metadata
                self.is_assera_variant = False
        if self.is_assera_variant:
            self.function_name = self.metadata["_function"]
            self.initialize_function_name = self.metadata["_initialize_function"]
            self.deinitialize_function_name = self.metadata["_deinitialize_function"]
            self.domain = self.metadata["domain"]
            self.custom_metadata = {
                key: self.metadata[key]
                for key in self.metadata if key not in variant_metadata_keys
            }


class AsseraLibraryData:
    def __init__(self, toml_document):
        self.library_toml = toml_document
        self.name = toml_document["library_name"]
        self.module_names = toml_document["module_names"]
        self.modules_table = toml_document["modules"]
        self.modules = []
        for module_name in self.modules_table:
            self.modules.append(AsseraModuleData(self.modules_table[module_name]))


def parse_toml_header(filepath):
    path = os.path.abspath(filepath)
    toml_doc = None
    with open(path, "r") as f:
        file_contents = f.read()
        toml_doc = tomlkit.parse(file_contents)
    return AsseraLibraryData(toml_doc)


def print_assera_toml_library_data(assera_library_data):
    print("Library Name: {}".format(assera_library_data.name))
    print("Module Names: {}".format(assera_library_data.module_names))
    for module in assera_library_data.modules:
        print("\tModule: {}".format(module.name))
        print("\t\tIs Assera Sample Variant: {}".format(module.is_assera_variant))
        if module.is_assera_variant:
            print("\t\tVariant Function: {}".format(module.function_name))
            print("\t\tInit Function: {}".format(module.initialize_function_name))
            print("\t\tDe-Init Function: {}".format(module.deinitialize_function_name))
            print("\t\tDomain: {}".format(module.domain))
            print("\t\tCustom Metadata Parameters:")
            for key in module.custom_metadata:
                print("\t\t\t{} : {}".format(key, module.custom_metadata[key]))
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to file to parse as TOML")
    args = parser.parse_args()

    assera_lib_data = parse_toml_header(args.input)
    print_assera_toml_library_data(assera_lib_data)


if __name__ == "__main__":
    main()
