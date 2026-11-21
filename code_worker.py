# -*- coding: utf-8 -*-
"""spawn worker: 只import p3code, 供ProcessPoolExecutor(spawn)调用"""
import p3code


def run(item):
    return p3code.code_np(item)
