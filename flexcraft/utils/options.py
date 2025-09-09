"""Utilities for option parsing scripts."""

import argparse

class OptionType:
  def __init__(self, default, parser=None, **kwargs):
    self.default = default
    self.type = parser or type(self.default)
    self.kwargs = kwargs

def option_parser(description, **kwargs):
  parser = argparse.ArgumentParser(description=description)
  for name in kwargs:
    default = kwargs[name]
    dtype = type(default)
    argparse_kwargs = {}
    if isinstance(default, OptionType):
      dtype = default.type
      argparse_kwargs = default.kwargs
      default = default.default
    elif isinstance(default, tuple):
      default, dtype = default
    parser.add_argument(
      f"--{name}", default=default,
      type=dtype, required=False,
      **argparse_kwargs
    )
  return parser

def parse_options(description, **kwargs):
  parser = option_parser(description, **kwargs)
  return parser.parse_args()
