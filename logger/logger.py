#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import logging.config
import json
import os


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_LOGGER_CONFIG = os.path.join(_CURRENT_DIR, "logger_config.json")


def setup_logging(save_dir, log_config=_DEFAULT_LOGGER_CONFIG, default_level=logging.INFO):
    log_config = os.path.abspath(log_config)
    if os.path.isfile(log_config):
        data = None
        with open(log_config) as json_file:
            data = json.load(json_file)
            if "filename" in data["handlers"].keys():
                data["handlers"]["filename"] = os.path.join(save_dir, data["handlers"]["filename"])
        logging.config.dictConfig(data)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
