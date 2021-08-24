# -*- coding: utf-8 -*-
"""Set of helpful functions.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
import datetime
import getpass
import platform


def get_runtime_user_information():
    r"""Get user runtime information.

    Returns
    -------
    RUI : dict
        Holding username in 'user', pc name in 'pc', date of execution 'date'.

    """
    # get time without microseconds
    date = datetime.datetime.now()
    date = date.isoformat(sep=' ', timespec='seconds')

    return {
        'user': getpass.getuser(),
        'pc': platform.node(),
        'date': date,
    }
