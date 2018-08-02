import functools
import tensorflow
import os
import sys

def doublewrap(function):
    """
    A decorator of decorator, allowing use of lazy property if no arguments are provided
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope = None, *args, **kwargs):
    """
    Lazy decorator, optimizes code by loading class Model parts only once to memory
    Also its groups tf.Graph, in tensorboard into smaller, more readable parts
    """    
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            #Sorting Graph by Var_scope
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
                print("Initialized Model.{}".format(name))
        return getattr(self, attribute)
    
    return decorator