# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tensorboard.compat."""

import importlib
import sys
import types
import unittest


_MISSING = object()


class CompatTest(unittest.TestCase):
    def setUp(self):
        self._saved_tensorflow = sys.modules.get("tensorflow", _MISSING)

    def tearDown(self):
        if self._saved_tensorflow is _MISSING:
            sys.modules.pop("tensorflow", None)
        else:
            sys.modules["tensorflow"] = self._saved_tensorflow
        import tensorboard.compat as compat

        importlib.reload(compat)

    def _reload_compat(self):
        import tensorboard.compat as compat

        return importlib.reload(compat)

    def testFallsBackToStubWhenTensorflowLacksRequiredApi(self):
        fake_tensorflow = types.ModuleType("tensorflow")
        fake_tensorflow.__version__ = "fake"
        sys.modules["tensorflow"] = fake_tensorflow

        compat = self._reload_compat()

        self.assertEqual(compat.tf.__version__, "stub")
        self.assertTrue(hasattr(compat.tf, "io"))
        self.assertTrue(hasattr(compat.tf.io, "gfile"))

    def testUsesTensorflowWhenRequiredApiIsPresent(self):
        fake_tensorflow = types.ModuleType("tensorflow")
        fake_tensorflow.__version__ = "fake"
        fake_tensorflow.compat = types.SimpleNamespace(v2=object())
        fake_tensorflow.errors = types.SimpleNamespace()
        fake_tensorflow.io = types.SimpleNamespace(gfile=object())
        sys.modules["tensorflow"] = fake_tensorflow

        compat = self._reload_compat()

        self.assertEqual(compat.tf.__version__, "fake")
        self.assertIs(compat.tf.io, fake_tensorflow.io)


if __name__ == "__main__":
    unittest.main()
