'''Every node exposed by H3-Extended is visibly marked experimental.'''

import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0]]


def load_extension_module():
    spec = importlib.util.spec_from_file_location(
        'h3_extended_display_test',
        PACK / '__init__.py',
        submodule_search_locations=[str(PACK)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_args = sys.argv[:]
    sys.argv = [previous_args[0], '--cpu']
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = previous_args
    return module


class ExperimentalDisplayNameTests(unittest.TestCase):
    def test_every_exposed_node_has_one_experimental_suffix(self):
        module = load_extension_module()
        nodes = asyncio.run(module.H3ExtendedExtension().get_node_list())
        self.assertTrue(nodes)
        for node in nodes:
            display_name = node.define_schema().display_name
            self.assertTrue(
                display_name.endswith(' (Experimental)'),
                '%s: %s' % (node.__name__, display_name),
            )
            self.assertFalse(
                display_name.endswith(
                    ' (Experimental) (Experimental)'
                )
            )


if __name__ == '__main__':
    unittest.main()
