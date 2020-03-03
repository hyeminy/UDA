from __future__ import absolute_import

from .resnet import *

__factory = {
    'resent50':resnet50
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
